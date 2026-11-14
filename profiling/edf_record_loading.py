"""Batch native EDF record access and compare CPU versus GPU window assembly.

edfio supplies validated signal/calibration metadata. Only the fixed EDF payload
layout (channel labels, samples per record and byte offsets) is interpreted here.
The original file is memory-mapped unchanged; no processed cache is created.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from profiling.native_edf_loading import NativeEdfReader
from profiling.signal_utils import calibrate


@dataclass(frozen=True)
class RecordLayout:
    """Offsets count int16 words, including annotation-channel payload words."""

    header_bytes: int
    record_count: int
    words_per_record: int
    samples_per_channel_record: int
    channel_offsets: np.ndarray


@dataclass(frozen=True)
class WindowPlan:
    """Validated requests and the unique source records needed by this batch."""

    starts: np.ndarray
    length: int
    record_ids: np.ndarray


def read_record_layout(path: Path, source: NativeEdfReader) -> RecordLayout:
    """Cross-check source payload offsets against edfio's public metadata.

    EDF stores all samples of each channel consecutively within each record.
    Annotation channels have their own word counts and must contribute to the
    stride even though they are not ordinary numerical output channels. Unknown
    record counts, trailing bytes, truncation and inconsistent layouts fail.
    """
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError("Expected a complete ordinary EDF header")
        header_bytes = int(fixed[184:192])
        record_count = int(fixed[236:244])
        channel_count = int(fixed[252:256])
        if channel_count < 1 or header_bytes != 256 * (channel_count + 1):
            raise ValueError("Unexpected EDF header length")
        fields = stream.read(header_bytes - 256)
    if len(fields) != header_bytes - 256:
        raise ValueError("Truncated EDF signal header")
    labels = [fields[index * 16 : (index + 1) * 16].decode("ascii").strip() for index in range(channel_count)]
    counts_start = 216 * channel_count
    counts = np.array(
        [int(fields[counts_start + index * 8 : counts_start + (index + 1) * 8]) for index in range(channel_count)],
        dtype=np.int64,
    )
    if record_count < 1 or np.any(counts < 1):
        raise ValueError("Known positive record and channel sample counts are required")
    numerical = np.array([label != "EDF Annotations" for label in labels])
    if tuple(label for label in labels if label != "EDF Annotations") != source.metadata.channel_names:
        raise ValueError("Source channel order disagrees with edfio")
    samples_per_record = counts[numerical]
    if not len(samples_per_record) or not np.all(samples_per_record == samples_per_record[0]):
        raise ValueError("Numerical channels require equal samples per record")
    if (
        header_bytes != source.recording.bytes_in_header_record
        or record_count != source.recording.num_data_records
        or record_count * int(samples_per_record[0]) != source.metadata.sample_count
    ):
        raise ValueError("Payload layout disagrees with edfio metadata")
    words_per_record = int(counts.sum())
    if path.stat().st_size != header_bytes + record_count * words_per_record * 2:
        raise ValueError("EDF file size does not match its declared record layout")
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))[numerical]
    return RecordLayout(header_bytes, record_count, words_per_record, int(samples_per_record[0]), offsets)


class BatchedEdfRecords:
    """Expose record-aware CPU and GPU paths with identical source byte offsets."""

    def __init__(self, path: Path, source: NativeEdfReader):
        self.metadata = source.metadata
        self.layout = read_record_layout(path, source)
        self.records = np.memmap(
            path,
            dtype="<i2",
            mode="r",
            offset=self.layout.header_bytes,
            shape=(self.layout.record_count, self.layout.words_per_record),
        )

    def plan(self, starts: np.ndarray, length: int) -> WindowPlan:
        """Validate before reading; preserve request order while deduplicating records."""
        if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer) or not len(starts):
            raise ValueError("Require a nonempty integer start vector")
        if length < 1 or np.any(starts < 0) or np.any(starts > self.metadata.sample_count - length):
            raise ValueError("Window extends outside the recording")
        samples_per_record = self.layout.samples_per_channel_record
        ranges = [
            np.arange(int(start) // samples_per_record, (int(start) + length - 1) // samples_per_record + 1)
            for start in starts
        ]
        return WindowPlan(starts.astype(np.int64, copy=True), length, np.unique(np.concatenate(ranges)))

    def read_cpu(self, plan: WindowPlan) -> np.ndarray:
        """Gather exactly requested int16 samples with broadcasted NumPy indexing.

        Index construction and output allocation occur per call. The file remains
        record-major/channel-blocked; no full recording is reshaped or copied.
        """
        times = plan.starts[:, None] + np.arange(plan.length)
        record_indices, within_record = np.divmod(times, self.layout.samples_per_channel_record)
        word_indices = within_record[:, :, None] + self.layout.channel_offsets
        return np.asarray(self.records[record_indices[:, :, None], word_indices])

    def read_records(self, plan: WindowPlan) -> np.ndarray:
        """Copy unique complete original records, retaining skipped annotation words."""
        return np.asarray(self.records[plan.record_ids])

    def read_cpu_from_records(self, plan: WindowPlan) -> np.ndarray:
        """Convert unique records once, then gather contiguous channel rows on CPU.

        Avoid a (batch, length, channels) int64 index tensor: extracting full
        channel blocks and making a time-major record array amortizes layout
        conversion over repeated windows and improves locality. Annotation gaps
        are removed using the same validated per-channel offsets.
        """
        raw = self.read_records(plan)
        channel_words = self.layout.channel_offsets[:, None] + np.arange(self.layout.samples_per_channel_record)
        time_major = np.ascontiguousarray(raw[:, channel_words].transpose(0, 2, 1))
        start_records, within_record = np.divmod(plan.starts, self.layout.samples_per_channel_record)
        local_starts = (
            np.searchsorted(plan.record_ids, start_records) * self.layout.samples_per_channel_record + within_record
        )
        channel_count = len(self.metadata.channel_names)
        flattened = time_major.reshape(-1, channel_count)
        output = np.empty((len(plan.starts), plan.length, channel_count), dtype=np.int16)
        # Every record within a requested window belongs to the sorted union.
        # Thus that window is contiguous in the compact time-major array, even
        # if gaps between different windows were removed from the source.
        for destination, start in zip(output, local_starts):
            np.copyto(destination, flattened[int(start) : int(start) + plan.length])
        return output

    def payload_statistics(self, plan: WindowPlan) -> dict:
        """Describe logical payload, record reuse and overread, not physical disk bytes."""
        times = plan.starts[:, None] + np.arange(plan.length)
        channel_count = len(self.metadata.channel_names)
        logical_bytes = len(plan.starts) * plan.length * channel_count * 2
        unique_signal_bytes = len(np.unique(times)) * channel_count * 2
        record_bytes = len(plan.record_ids) * self.layout.words_per_record * 2
        return {
            "unique_records": len(plan.record_ids),
            "logical_window_bytes": logical_bytes,
            "unique_requested_signal_bytes": unique_signal_bytes,
            "record_payload_bytes": record_bytes,
            "record_bytes_divided_by_logical_bytes": record_bytes / logical_bytes,
            "record_overread_vs_unique_signal_bytes": record_bytes / unique_signal_bytes,
        }


class GpuRecordAssembler:
    """Transfer native records and perform window/layout selection plus calibration on GPU.

    No source samples are read directly by the GPU. Metadata/calibration reside
    on the device once; each batch transfers its record IDs, starts and raw
    int16 records. Index construction and output allocation are included.
    """

    def __init__(self, source: BatchedEdfRecords, device: torch.device):
        self.source = source
        self.device = device
        self.channel_offsets = torch.from_numpy(source.layout.channel_offsets).to(device)
        self.gain = torch.from_numpy(source.metadata.gain).to(device)
        self.offset = torch.from_numpy(source.metadata.offset).to(device)

    def assemble(self, plan: WindowPlan) -> torch.Tensor:
        """Include source record reads, pinning, transfer, indexing and calibration."""
        raw = torch.from_numpy(self.source.read_records(plan)).pin_memory().to(self.device, non_blocking=True)
        record_ids = torch.from_numpy(plan.record_ids).to(self.device)
        starts = torch.from_numpy(plan.starts).to(self.device)
        samples_per_record = self.source.layout.samples_per_channel_record
        start_records = torch.div(starts, samples_per_record, rounding_mode="floor")
        local_starts = torch.searchsorted(record_ids, start_records) * samples_per_record + starts % samples_per_record
        times = local_starts[:, None] + torch.arange(plan.length, device=self.device)
        records = torch.div(times, samples_per_record, rounding_mode="floor")
        within_record = times % samples_per_record
        indices = (records * self.source.layout.words_per_record + within_record)[:, :, None] + self.channel_offsets
        digital = raw.flatten()[indices]
        return calibrate(digital, self.gain, self.offset)
