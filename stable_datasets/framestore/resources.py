"""How much of a node a job may use, and holding it to that. A node runs one job at a time (converting a store, or
training), so a job takes a configured share of the node -- CPUs, memory, GPU memory -- leaving the rest so the node
stays responsive; the container's limits (cgroup v1 or v2) count, not the machine's. The hard ceiling is the
platform's (container or scheduler memory limit); this keeps a job under it instead of being killed at it.

    r = Resources(cpu=0.8, memory="16GB", gpu_memory=0.8)  # fractions of what is free, or absolute amounts
    b = Budget("convert", r)                                # logs what it found and what it allows
    b.reserve("lance scan", nbytes)                         # a fixed need, from a measured or known size
    with b.measure("GPU decoder"):                          # charge what the block actually took
        dec = make_decoder()
    workers, slots = b.allocate([Stage("workers", 8, per_worker), Stage("slots", 16, per_slot, least=2, shm=True)])
    gate = b.gate("batches in flight")                      # the rest bounds the bytes a pipeline holds at once
    gate.acquire(n); ...; gate.release(n)

allocate() gives every stage its minimum before any stage more than that, then grows them in turn, and raises a
ResourceError naming the stage when the minimums do not fit. A gate blocks a producer while the bytes in flight
would exceed its limit (one item always passes when nothing is in flight), so the pipeline's memory follows the
items' real sizes, whatever the frame or page size. Messages go to the "resources" logger.
"""

import contextlib
import dataclasses
import logging
import os
import re
import shutil
import subprocess
import threading

from ._native import usable_cpus


GB = 1e9
SPARE_CPUS = 2  # never handed to a job, whatever it asks: a node whose job keeps every CPU busy can stop responding


def job_cpus(want=1.0):
    """CPUs a job may keep busy: `want` (a fraction or a count) of the CPUs the container may use, leaving
    SPARE_CPUS of them free (at least one for the job)."""
    n = usable_cpus()
    return max(1, min(int(resolve(want, n)), n - SPARE_CPUS))


log = logging.getLogger("resources")


class ResourceError(RuntimeError):
    pass


def gb(n):
    return f"{n / GB:.1f} GB"


def parse_amount(v):
    """A fraction (0 < v <= 1) of what is available, or an absolute amount: a count (cpu=6) or a size ("16GB",
    "512MiB"). Returns ("fraction", f) or ("absolute", n)."""
    if isinstance(v, str):
        m = re.fullmatch(r"\s*([\d.]+)\s*([kmgt]?)(i?b?)\s*", v.lower())
        if not m:
            raise ValueError(f"cannot parse amount {v!r} (a fraction like 0.8, a count, or a size like 16GB)")
        x, unit, ib = float(m.group(1)), m.group(2), m.group(3)
        if not unit and not ib:  # as a number: "1" is one (a count), "1.0" all of it (a fraction)
            return parse_amount(x if "." in m.group(1) else int(x))
        return "absolute", x * (1024 if ib.startswith("i") else 1000) ** "_kmgt".index(unit or "_")
    if 0 < v <= 1 and isinstance(v, float):
        return "fraction", v
    if v > 0:
        return "absolute", v
    raise ValueError(f"amount must be positive, got {v!r}")


def resolve(v, available):
    kind, x = parse_amount(v)
    return x * available if kind == "fraction" else min(x, available)


@dataclasses.dataclass
class Resources:
    """The share of the node a job may use. Each of cpu, memory and gpu_memory is a fraction of what is available
    when the job starts (CPUs within the container's quota, free memory, free GPU memory) or an absolute amount
    (a CPU count; a size such as "16GB"). nice lowers CPU and I/O priority (0: unchanged)."""

    cpu: object = 0.8
    memory: object = 0.8
    gpu_memory: object = 0.8
    nice: int = 0

    @classmethod
    def add_arguments(cls, ap, prefix="", **defaults):
        d = dataclasses.replace(cls(), **defaults)
        for f in ("cpu", "memory", "gpu_memory"):
            ap.add_argument(
                f"--{prefix}{f.replace('_', '-')}",
                default=getattr(d, f),
                help=f"{f.replace('_', ' ')} to use: a fraction of what is free (default %(default)s), "
                f"or an amount ({'a count' if f == 'cpu' else 'e.g. 16GB'})",
            )
        ap.add_argument(f"--{prefix}nice", type=int, default=d.nice, help="lower CPU and I/O priority by this much")

    @classmethod
    def from_args(cls, a, prefix=""):
        p = prefix.replace("-", "_")
        return cls(*(_number(getattr(a, p + f)) for f in ("cpu", "memory", "gpu_memory")), getattr(a, p + "nice"))


def _number(v):
    try:
        return float(v) if "." in str(v) else int(v)
    except ValueError:
        return v  # a size such as "16GB"


# ---------------------------------------------------------------------------------------------------------------
# what the node has
# ---------------------------------------------------------------------------------------------------------------
def _int(path):
    try:
        v = open(path).read().strip()
    except OSError:
        return None
    return None if v in ("max", "") else int(v)


def _stat(path, key):
    try:
        for line in open(path):
            k, v = line.split()[:2]
            if k == key:
                return int(v)
    except OSError:
        pass
    return 0


def memory_available():
    """Bytes this process can still allocate without swapping or the OOM killer: the machine's available memory
    (free + reclaimable page cache), capped by the container's limit minus its usage (page cache it could drop
    not counted as used). Shared memory (/dev/shm) counts as used here."""
    if os.path.exists("/proc/meminfo"):
        avail = _stat("/proc/meminfo", "MemAvailable:") * 1024
    else:  # macOS: free, inactive and purgeable pages
        try:
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(vm.split("page size of ")[1].split()[0])
            pages = {
                k.strip(): int(v.strip().rstrip("."))
                for k, v in (line.split(":") for line in vm.splitlines()[1:] if ":" in line)
            }
            avail = page * sum(
                pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            )
        except (OSError, IndexError, ValueError):
            avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    for limit, usage, stat, inactive in (
        (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.stat",
            "inactive_file",
        ),  # cgroup v2
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.stat",
            "total_inactive_file",
        ),
    ):  # cgroup v1
        lim = _int(limit)
        if lim is None or lim >= 1 << 60:  # v1 writes "no limit" as ~2^63
            continue
        used = (_int(usage) or 0) - _stat(stat, inactive)
        avail = min(avail, max(0, lim - used))
        break
    return avail


def shm_free(path="/dev/shm"):
    """Free bytes in /dev/shm, where torch's shared tensors and the frame index live on Linux (containers often
    mount only 64 MB there); None where it is not a separate filesystem (macOS)."""
    return shutil.disk_usage(path).free if os.path.isdir(path) else None


def gpu_free(device=0):
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info(device)[0]


def process_bytes(pid=None):
    """A process's private resident memory (anonymous pages: not its shared-memory tensors or mapped files), or
    None where /proc is missing."""
    try:
        for line in open(f"/proc/{pid or 'self'}/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def peak_process_bytes():
    """This process's peak resident memory."""
    import resource
    import sys

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def limit_malloc_arenas(n=2):
    """Cap glibc's malloc arenas (its default: up to 8 per CPU, one claimed by each new thread) for this process's
    threads from now on. Freed memory stays in each arena, so a converter's 18 threads each kept up to ~0.2 GB of
    batch-sized buffers: with --memory 4GB peak memory was 7.0 GB, 5.3 GB with 2 arenas (and 6% faster). Call
    it before starting threads. MALLOC_ARENA_MAX in the environment wins; no-op without glibc."""
    if os.environ.get("MALLOC_ARENA_MAX"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").mallopt(-8, n)  # M_ARENA_MAX
    except (OSError, AttributeError):
        pass


def lower_priority(nice=10):
    """Run below interactive work (CPU and, on Linux, I/O priority), so a long batch job keeps the node usable;
    it still gets every idle cycle. Call it before starting threads: Linux keeps both per thread, and new threads
    inherit their creator's."""
    if nice:
        os.nice(nice)
        if shutil.which("ionice"):
            subprocess.run(["ionice", "-c2", "-n7", "-p", str(os.getpid())], capture_output=True)


# ---------------------------------------------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Stage:
    """A stage sized in units of `unit` bytes: up to `want` of them, at least `least`. shm: also counted against
    /dev/shm; gpu: GPU memory instead of host memory."""

    name: str
    want: int
    unit: float
    least: int = 1
    shm: bool = False
    gpu: bool = False


class ByteGate:
    """Bounds the bytes a pipeline holds at once: acquire(n) blocks while n more would exceed the limit, except
    when nothing is in flight (an item larger than the limit still passes, alone)."""

    def __init__(self, name, limit):
        self.name, self.limit = name, max(1, int(limit))
        self.used = self.peak = 0
        self.cond = threading.Condition()

    def acquire(self, n):
        n = int(n)
        with self.cond:
            while self.used and self.used + n > self.limit:
                self.cond.wait()
            self.used += n
            self.peak = max(self.peak, self.used)
        return n

    def try_acquire(self, n):
        """acquire(n) if it would not block; returns whether it did."""
        with self.cond:
            if self.used and self.used + int(n) > self.limit:
                return False
            self.used += int(n)
            self.peak = max(self.peak, self.used)
        return True

    def release(self, n):
        with self.cond:
            self.used -= int(n)
            self.cond.notify_all()


class Budget:
    """Bytes a job may take, by pool: host memory (shared memory is part of it, and is also limited by /dev/shm's
    own size) and GPU memory (read when first needed); and the CPUs it may keep busy (job_cpus: SPARE_CPUS stay
    free)."""

    def __init__(self, what, resources=None):
        r = resources or Resources()
        lower_priority(r.nice)
        self.what, self.r = what, r
        self.cpus = job_cpus(r.cpu)
        if self.cpus < int(resolve(r.cpu, usable_cpus())):
            log.info(f"BUDGET {what}: {SPARE_CPUS} CPUs stay free for the node: {self.cpus} instead of asked {r.cpu}")
        self.available = memory_available()
        self.memory = resolve(r.memory, self.available)
        self.start = process_bytes() or 0
        self.charged = 0  # host bytes of this process already charged (measure, charge_process)
        self.left = {"memory": self.memory, "shm": shm_free(), "gpu": None}
        shm = self.left["shm"]
        log.info(
            f"BUDGET {what}: {self.cpus} of {usable_cpus()} CPUs, {gb(self.memory)} of {gb(self.available)} "
            f"free memory" + (f", {gb(shm)} free in /dev/shm" if shm is not None else "")
        )

    def _left(self, pool):
        if pool == "gpu" and self.left["gpu"] is None:
            free = gpu_free()
            if free is None:
                raise ResourceError(f"{self.what}: no CUDA device")
            self.gpu = self.left["gpu"] = resolve(self.r.gpu_memory, free)
            log.info(f"BUDGET {self.what}: {gb(self.gpu)} of {gb(free)} free GPU memory")
        return self.left[pool]

    def _pools(self, s):
        return ["gpu"] if s.gpu else ["memory"] + (["shm"] if s.shm and self.left["shm"] is not None else [])

    def allocate(self, stages):
        """Counts for the stages: every stage's minimum first, then one more unit to each in turn while it fits."""
        left = {p: self._left(p) for s in stages for p in self._pools(s)}
        n = [s.least for s in stages]
        for p in left:
            need = sum(s.least * s.unit for s in stages if p in self._pools(s))
            if need > left[p]:
                names = ", ".join(f"{s.least} x {s.name} ({gb(s.unit)})" for s in stages if p in self._pools(s))
                raise ResourceError(
                    f"{self.what}: {names} need {gb(need)} of {p}, but the budget has {gb(left[p])} "
                    f"left. Free {p}, use smaller batches, or give the job more (--memory, "
                    f"--gpu-memory)"
                )
            left[p] -= need
        grew = True
        while grew:
            grew = False
            for i, s in enumerate(stages):
                if n[i] < s.want and all(left[p] >= s.unit for p in self._pools(s)):
                    n[i] += 1
                    for p in self._pools(s):
                        left[p] -= s.unit
                    grew = True
        for s, k in zip(stages, n):
            for p in self._pools(s):
                self.left[p] -= k * s.unit
            if k < s.want:
                log.info(f"BUDGET {self.what}: {s.name}: {k} instead of {s.want} ({gb(s.unit)} each)")
        return n

    def reserve(self, name, nbytes, shm=False, gpu=False):
        self.allocate([Stage(name, 1, nbytes, shm=shm, gpu=gpu)])

    def release(self, nbytes, shm=False, gpu=False):
        for p in self._pools(Stage("", 0, 0, shm=shm, gpu=gpu)):
            self.left[p] += nbytes

    @contextlib.contextmanager
    def measure(self, name, gpu=False):
        """Charge the host (and with gpu=True, GPU) memory the block took; refuses after it if that did not fit."""
        h0, g0 = process_bytes(), gpu_free() if gpu else None
        yield
        h1, g1 = process_bytes(), gpu_free() if gpu else None
        if h0 is not None and h1 is not None and h1 > h0:
            self.reserve(f"{name} (measured)", h1 - h0)
            self.charged += h1 - h0
        if g0 is not None and g1 is not None and g0 > g1:
            self._left("gpu")
            self.reserve(f"{name} (measured, GPU)", g0 - g1, gpu=True)

    def charge_process(self, name):
        """Charge what this process holds now beyond what it held when the budget was made and what was already
        charged (imports, a CUDA context, libraries' buffers): call it once set up, before sizing what remains."""
        now = process_bytes()
        if now is not None and now - self.start > self.charged:
            n = now - self.start - self.charged
            self.reserve(name, n)
            self.charged += n
            log.info(f"BUDGET {self.what}: {name}: {gb(n)}")

    def gate(self, name, least):
        """A ByteGate over the rest of the host budget; refuses if that is less than `least` bytes."""
        if self.left["memory"] < least:
            raise ResourceError(
                f"{self.what}: {name} needs at least {gb(least)}, but the budget has "
                f"{gb(self.left['memory'])} left. Give the job more memory (--memory)"
            )
        g = ByteGate(name, self.left["memory"])
        self.left["memory"] = 0
        log.info(f"BUDGET {self.what}: {name}: up to {gb(g.limit)}")
        return g

    def report(self):
        """Peak memory against the budget (this process's growth since the budget was made)."""
        return (
            f"peak memory {gb(peak_process_bytes())} ({gb(self.start)} before the budget, which allowed "
            f"{gb(self.memory)} more)"
        )
