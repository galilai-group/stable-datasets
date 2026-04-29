from __future__ import annotations

import cProfile
import json
import pstats
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil


def total_host_uss_bytes(process: psutil.Process | None = None) -> int:
    process = process or psutil.Process()
    total = 0
    targets = [process]

    try:
        targets.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

    for proc in targets:
        try:
            total += proc.memory_full_info().uss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class PeakMemoryMonitor:
    """Background sampler for total host USS across parent and children."""

    def __init__(self, poll_interval_sec: float = 0.05):
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_uss_bytes = 0

    @property
    def peak_uss_bytes(self) -> int:
        return self._peak_uss_bytes

    def _sample_forever(self) -> None:
        while not self._stop_event.is_set():
            self._peak_uss_bytes = max(self._peak_uss_bytes, total_host_uss_bytes())
            self._stop_event.wait(self.poll_interval_sec)

    def start(self) -> None:
        self._peak_uss_bytes = max(self._peak_uss_bytes, total_host_uss_bytes())
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_forever, name="peak-uss-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5 * self.poll_interval_sec))
        self._peak_uss_bytes = max(self._peak_uss_bytes, total_host_uss_bytes())
        return self._peak_uss_bytes


def measure_peak_uss_during(fn, *, poll_interval_sec: float = 0.05):
    monitor = PeakMemoryMonitor(poll_interval_sec=poll_interval_sec)
    monitor.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        peak_uss_bytes = monitor.stop()
    elapsed = time.perf_counter() - started
    return result, peak_uss_bytes, elapsed


@dataclass
class StageProfile:
    enabled: bool = False
    loader_next_calls: int = 0
    loader_next_total_sec: float = 0.0
    source_getitem_calls: int = 0
    source_getitem_total_sec: float = 0.0
    source_getitems_calls: int = 0
    source_getitems_total_sec: float = 0.0
    backend_take_calls: int = 0
    backend_take_total_sec: float = 0.0
    format_batch_calls: int = 0
    format_batch_total_sec: float = 0.0
    normalize_calls: int = 0
    normalize_total_sec: float = 0.0
    collate_calls: int = 0
    collate_total_sec: float = 0.0
    profiled_batches: int = 0
    profiled_images: int = 0
    notes: list[str] = field(default_factory=list)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def record_loader_next(self, elapsed_sec: float, batch_size: int) -> None:
        if not self.enabled:
            return
        self.loader_next_calls += 1
        self.loader_next_total_sec += elapsed_sec
        self.profiled_batches += 1
        self.profiled_images += batch_size

    def record_source_getitem(self, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        self.source_getitem_calls += 1
        self.source_getitem_total_sec += elapsed_sec

    def record_source_getitems(self, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        self.source_getitems_calls += 1
        self.source_getitems_total_sec += elapsed_sec

    def record_backend_take(self, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        self.backend_take_calls += 1
        self.backend_take_total_sec += elapsed_sec

    def record_format_batch(self, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        self.format_batch_calls += 1
        self.format_batch_total_sec += elapsed_sec

    def record_normalize(self, elapsed_sec: float, count: int = 1) -> None:
        if not self.enabled:
            return
        self.normalize_calls += count
        self.normalize_total_sec += elapsed_sec

    def record_collate(self, elapsed_sec: float) -> None:
        if not self.enabled:
            return
        self.collate_calls += 1
        self.collate_total_sec += elapsed_sec

    def summary(self) -> dict:
        source_total = self.source_getitem_total_sec + self.source_getitems_total_sec
        unattributed = self.loader_next_total_sec - source_total - self.normalize_total_sec - self.collate_total_sec
        return {
            **asdict(self),
            "source_fetch_total_sec": source_total,
            "unattributed_loader_overhead_sec": max(0.0, unattributed),
            "loader_next_avg_sec": (
                self.loader_next_total_sec / self.loader_next_calls if self.loader_next_calls else None
            ),
            "backend_take_avg_sec": (
                self.backend_take_total_sec / self.backend_take_calls if self.backend_take_calls else None
            ),
            "format_batch_avg_sec": (
                self.format_batch_total_sec / self.format_batch_calls if self.format_batch_calls else None
            ),
            "normalize_avg_sec_per_call": (
                self.normalize_total_sec / self.normalize_calls if self.normalize_calls else None
            ),
            "collate_avg_sec_per_call": (self.collate_total_sec / self.collate_calls if self.collate_calls else None),
        }


@contextmanager
def timed(callback):
    started = time.perf_counter()
    try:
        yield
    finally:
        callback(time.perf_counter() - started)


def dump_stage_profile(profile: StageProfile, output_path: str | Path, metadata: dict | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "stages": profile.summary(),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def dump_cprofile(
    profiler: cProfile.Profile,
    *,
    pstats_path: str | Path,
    text_path: str | Path,
    sort_by: str = "cumtime",
    lines: int = 40,
) -> None:
    pstats_path = Path(pstats_path)
    text_path = Path(text_path)
    pstats_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    profiler.dump_stats(str(pstats_path))
    with text_path.open("w") as handle:
        stats = pstats.Stats(profiler, stream=handle)
        stats.strip_dirs().sort_stats(sort_by).print_stats(lines)
