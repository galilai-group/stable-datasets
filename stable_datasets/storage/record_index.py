"""A per-node index of a store's records: where each record's bytes are and what a reader needs to address its
units, built once and memory-mapped by every loader process on the node (one copy in RAM, not one per worker).

    idx = RecordIndex.shared(key, build, nbytes)      # opened if this node has it, else built by the first process
    c = idx.lookup(rows)                              # every column at rows, plus the rows' ragged values

Layout (a directory):
    meta.json             version, rows, files (fid indexes this list), columns, ragged column, caller's fields
    <column>.npy          one value per record (e.g. fid, pos, size, h, w, mcu_h, mcu_w)
    <ragged>.bin          optional: a variable number of values per record, back to back (e.g. MCU segment lengths)
    <ragged>_off.npy      rows + 1 offsets into it

Where: $SDS_INDEX_DIR (or $FRAMESTORE_INDEX_DIR) if set, else /dev/shm when the index takes at most half of both the
free shared memory and the free memory (/dev/shm is RAM, and containers often mount only 64 MB of it), else local
disk under ~/.cache/stable_datasets/index, where the page cache keeps what loaders read hot.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping

import numpy as np


VERSION = 2
SHM_BASE = "/dev/shm/sds_index"
DISK_BASE = os.path.expanduser("~/.cache/stable_datasets/index")


def env_base() -> str | None:
    return os.environ.get("SDS_INDEX_DIR") or os.environ.get("FRAMESTORE_INDEX_DIR")


def index_base(nbytes: int) -> str:
    """Where to build an index of nbytes: the environment's choice, else /dev/shm if it fits comfortably, else disk."""
    if env_base():
        return env_base()
    from ..framestore.resources import gb, memory_available, shm_free

    shm = shm_free()
    if shm is not None and nbytes <= 0.5 * min(shm, memory_available()):
        return SHM_BASE
    if shm is not None:
        print(
            f"INDEX {gb(nbytes)} would take over half of the free shared memory ({gb(shm)}) or memory "
            f"({gb(memory_available())}): building it on disk under {DISK_BASE}",
            flush=True,
        )
    return DISK_BASE


class RecordIndex:
    def __init__(self, root: str):
        with open(f"{root}/meta.json") as f:
            meta = json.load(f)
        if meta.get("version") != VERSION:
            raise ValueError(f"index {root} has format {meta.get('version')}, expected {VERSION}")
        self.root, self.meta, self.files = root, meta, meta["files"]
        self.columns = {k: np.load(f"{root}/{k}.npy", mmap_mode="r") for k in meta["columns"]}
        self.ragged_name = meta.get("ragged")
        if self.ragged_name:
            self.ragged_off = np.load(f"{root}/{self.ragged_name}_off.npy", mmap_mode="r")
            dtype = np.dtype(meta["ragged_dtype"])
            self.ragged = (
                np.memmap(f"{root}/{self.ragged_name}.bin", dtype, mode="r")
                if self.ragged_off[-1]
                else np.zeros(0, dtype)
            )

    def __len__(self) -> int:
        return self.meta["rows"]

    def __getattr__(self, name):
        columns = self.__dict__.get("columns", {})
        if name in columns:
            return columns[name]
        raise AttributeError(name)

    def lookup(self, rows, gather: bool = True) -> dict[str, np.ndarray]:
        """Every column at rows. With a ragged column: ``n`` (values per row) and, gathered, the rows' values back
        to back with ``<ragged>_start`` each row's first; ``gather=False`` gives the whole ragged array without a
        copy and each row's start in it."""
        rows = np.asarray(rows, np.int64)
        out = {k: np.asarray(v[rows]) for k, v in self.columns.items()}
        if self.ragged_name:
            a, b = self.ragged_off[rows], self.ragged_off[rows + 1]
            n = b - a
            out["n"] = n
            if gather:
                idx = np.repeat(a - np.cumsum(n) + n, n) + np.arange(int(n.sum()))
                out[self.ragged_name] = self.ragged[idx]
                out[f"{self.ragged_name}_start"] = np.cumsum(n) - n
            else:
                out[self.ragged_name] = np.asarray(self.ragged)
                out[f"{self.ragged_name}_start"] = a
        return out

    @staticmethod
    def write(
        root: str,
        files: list[str],
        columns: Mapping[str, np.ndarray],
        *,
        ragged: tuple[str, np.ndarray, np.ndarray] | None = None,
        **fields,
    ) -> None:
        """Write an index to ``root`` (through root.tmp.<pid>, renamed when complete). ``ragged``: (name, offsets
        of rows + 1, values) or (name, offsets, path of a file of the values to move in)."""
        lengths = {len(v) for v in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"index columns of different lengths: {sorted(lengths)}")
        rows = lengths.pop() if lengths else 0
        tmp = f"{root}.tmp.{os.getpid()}"
        os.makedirs(tmp)
        try:
            for k, v in columns.items():
                np.save(f"{tmp}/{k}.npy", np.asarray(v))
            meta = {"version": VERSION, "rows": int(rows), "files": list(files), "columns": list(columns), **fields}
            if ragged is not None:
                name, off, values = ragged
                off = np.asarray(off, np.int64)
                if len(off) != rows + 1:
                    raise ValueError("ragged offsets need rows + 1 entries")
                np.save(f"{tmp}/{name}_off.npy", off)
                if isinstance(values, str | os.PathLike):
                    dtype = meta.get("ragged_dtype")
                    if dtype is None:
                        raise ValueError("a ragged column read from a file needs ragged_dtype=")
                    shutil.move(os.fspath(values), f"{tmp}/{name}.bin")
                else:
                    values = np.ascontiguousarray(values)
                    values.tofile(f"{tmp}/{name}.bin")
                    dtype = values.dtype.str
                meta.update(ragged=name, ragged_dtype=dtype)
            with open(f"{tmp}/meta.json", "w") as f:
                json.dump(meta, f)
            os.rename(tmp, root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @classmethod
    def shared(
        cls, key: str, build: Callable[[str], None], nbytes: int, *, base: str | None = None, owner: str | None = None
    ) -> RecordIndex:
        """The node's index named by ``key`` (a string that changes whenever the indexed data does): opened if
        present, else ``build(root)`` by the first process to get here while the others wait. ``owner`` (e.g. the
        store's path) marks the index so a new build removes the owner's older ones."""
        name = hashlib.sha1(key.encode()).hexdigest()[:16]
        for b in [base] if base else [env_base(), SHM_BASE, DISK_BASE]:
            if b and _current(f"{b}/{name}"):
                return cls(f"{b}/{name}")
        base = base or index_base(nbytes)
        os.makedirs(base, exist_ok=True)
        if shutil.disk_usage(base).free < 1.1 * nbytes:
            raise RuntimeError(
                f"the index needs ~{nbytes / 1e9:.1f} GB; {base} has "
                f"{shutil.disk_usage(base).free / 1e9:.1f} GB free (set $SDS_INDEX_DIR elsewhere)"
            )
        root = f"{base}/{name}"
        if not _current(root):
            with open(f"{root}.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not _current(root):
                    shutil.rmtree(root, ignore_errors=True)  # an index of an older format
                    t0 = time.time()
                    build(root)
                    _add_fields(root, build_s=round(time.time() - t0, 1), owner=owner)
                    if owner is not None:
                        remove_stale(owner, root)
        return cls(root)


def _current(root: str) -> bool:
    try:
        with open(f"{root}/meta.json") as f:
            return json.load(f).get("version") == VERSION
    except (OSError, ValueError):
        return False


def _add_fields(root: str, **fields) -> None:
    with open(f"{root}/meta.json") as f:
        meta = json.load(f)
    meta.update(fields)
    with open(f"{root}/meta.json.tmp", "w") as f:
        json.dump(meta, f)
    os.replace(f"{root}/meta.json.tmp", f"{root}/meta.json")


def remove_stale(owner: str, keep: str) -> None:
    """Remove the node's other indexes of ``owner`` (older versions of it), which would otherwise fill /dev/shm."""
    import glob

    for b in {env_base(), SHM_BASE, DISK_BASE} - {None}:
        for meta in glob.glob(f"{b}/*/meta.json"):
            root = os.path.dirname(meta)
            try:
                with open(meta) as f:
                    same = json.load(f).get("owner") == owner
            except (OSError, ValueError):
                continue
            if same and os.path.abspath(root) != os.path.abspath(keep):
                shutil.rmtree(root, ignore_errors=True)
                if os.path.exists(f"{root}.lock"):
                    os.remove(f"{root}.lock")
