"""Validate native record offsets, annotation strides and CPU/GPU assembly."""

import importlib

import numpy as np
import pytest
import torch


mne = pytest.importorskip("mne")
pytest.importorskip("edfio")
pytest.importorskip("pyedflib")
native = importlib.import_module("profiling.edf_source")
records_module = importlib.import_module("profiling.edf_record_loading")
reference_checks = importlib.import_module("profiling.edf_reference")
calibration = importlib.import_module("profiling.signal_calibration")


def make_source(path, *, annotation_in_middle=False):
    """Write known native samples, optionally relocating the annotation channel.

    Reordering is fixture construction only: header fields and corresponding
    payload blocks move together, testing a layout that fixed channel strides
    would decode incorrectly.
    """
    from pyedflib import highlevel

    samples = np.arange(3 * 384, dtype=np.int32).reshape(3, 384) - 500
    headers = [
        highlevel.make_signal_header(f"EEG{channel}", sample_frequency=128, physical_min=-100, physical_max=300)
        for channel in range(3)
    ]
    header = highlevel.make_header()
    header["annotations"] = [[0.5, 0.25, "test-event"]]
    highlevel.write_edf(str(path), samples, headers, header, digital=True)
    if annotation_in_middle:
        content = path.read_bytes()
        fixed = content[:256]
        channel_count = int(fixed[252:256])
        header_bytes = int(fixed[184:192])
        record_count = int(fixed[236:244])
        fields = content[256:header_bytes]
        order = [0, 3, 1, 2]
        reordered = []
        position = 0
        for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
            field = np.frombuffer(fields[position : position + width * channel_count], dtype=np.uint8)
            reordered.append(field.reshape(channel_count, width)[order].tobytes())
            position += width * channel_count
        counts = [
            int(fields[216 * channel_count + i * 8 : 216 * channel_count + (i + 1) * 8]) for i in range(channel_count)
        ]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        raw = np.frombuffer(content[header_bytes:], dtype="<i2").reshape(record_count, sum(counts))
        payload = np.concatenate([raw[:, offsets[i] : offsets[i + 1]] for i in order], axis=1)
        path.write_bytes(fixed + b"".join(reordered) + payload.tobytes())
    return samples.T.astype(np.int16)


@pytest.mark.parametrize("annotation_in_middle", [False, True])
def test_vectorized_layouts_match_native_samples_and_mne(tmp_path, annotation_in_middle):
    path = tmp_path / "source.edf"
    samples = make_source(path, annotation_in_middle=annotation_in_middle)
    before = path.read_bytes()
    source = native.NativeEdfReader(path)
    records = records_module.BatchedEdfRecords(path, source)
    starts = np.array([127, 0, 256, 127])
    plan = records.plan(starts, 128)
    expected = np.stack([samples[start : start + 128] for start in starts])
    np.testing.assert_array_equal(source.read_windows(starts, 128), expected)
    np.testing.assert_array_equal(records.read_cpu_from_records(plan), expected)
    np.testing.assert_array_equal(plan.record_ids, [0, 1, 2])
    gapped = records.plan(np.array([256, 0, 256]), 16)
    np.testing.assert_array_equal(gapped.record_ids, [0, 2])
    np.testing.assert_array_equal(
        records.read_cpu_from_records(gapped), np.stack([samples[start : start + 16] for start in gapped.starts])
    )
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as reference:
        physical = reference_checks.read_mne_windows(reference, starts, 128)
        reference_checks.validate_source(source, reference, starts, 128)
        calibrated = calibration.calibrate_cpu(
            records.read_cpu_from_records(plan), source.metadata.gain, source.metadata.offset
        )
        reference_checks.require_matching_signals(calibrated, physical)
    assert path.read_bytes() == before
    if annotation_in_middle:
        assert records.layout.channel_offsets[1] > records.layout.samples_per_channel_record


@pytest.mark.parametrize(
    "starts,length",
    [
        (np.array([-1]), 128),
        (np.array([257]), 128),
        (np.array([], dtype=np.int64), 128),
        (np.array([0.5]), 128),
        (np.array([0]), 0),
    ],
)
def test_invalid_plans_are_rejected(tmp_path, starts, length):
    path = tmp_path / "source.edf"
    make_source(path)
    records = records_module.BatchedEdfRecords(path, native.NativeEdfReader(path))
    with pytest.raises(ValueError):
        records.plan(starts, length)


def test_source_size_and_unknown_record_count_fail_before_mapping(tmp_path):
    path = tmp_path / "source.edf"
    make_source(path)
    source = native.NativeEdfReader(path)
    content = path.read_bytes()
    path.write_bytes(content[:-2])
    with pytest.raises(ValueError, match="file size"):
        records_module.BatchedEdfRecords(path, source)
    unknown = bytearray(content)
    unknown[236:244] = b"-1      "
    path.write_bytes(unknown)
    with pytest.raises(ValueError, match="Known positive"):
        records_module.BatchedEdfRecords(path, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_gpu_assembles_records_with_annotations_and_repeated_windows(tmp_path):
    path = tmp_path / "source.edf"
    samples = make_source(path, annotation_in_middle=True)
    before = path.read_bytes()
    source = native.NativeEdfReader(path)
    records = records_module.BatchedEdfRecords(path, source)
    assembler = records_module.GpuRecordAssembler(records, torch.device("cuda"))
    for starts, length in ((np.array([127, 0, 256, 127]), 128), (np.array([256, 0, 256]), 16), (np.array([0]), 384)):
        plan = records.plan(starts, length)
        actual = assembler.assemble(plan).cpu().numpy()
        expected = calibration.calibrate_cpu(
            np.stack([samples[start : start + length] for start in plan.starts]),
            source.metadata.gain,
            source.metadata.offset,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10)
        cpu = calibration.calibrate_cpu(
            records.read_cpu_from_records(plan), source.metadata.gain, source.metadata.offset
        )
        reference_checks.require_matching_signals(actual, cpu)
        with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as raw:
            reference_checks.require_matching_signals(actual, reference_checks.read_mne_windows(raw, starts, length))
    assert path.read_bytes() == before


def test_window_reader_cpu_matches_independent_mne_windows(tmp_path):
    path = tmp_path / "source.edf"
    make_source(path, annotation_in_middle=True)
    reader = records_module.EdfWindowReader(path)
    starts = np.array([127, 0, 256, 127])
    actual = reader.read(starts, 128)
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as raw:
        expected = reference_checks.read_mne_windows(raw, starts, 128)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=1e-10)
    with pytest.raises(ValueError, match="CUDA"):
        records_module.EdfWindowReader(path, "edf_gpu_assembly", torch.device("cpu"))
    with pytest.raises(ValueError, match="Unknown"):
        records_module.EdfWindowReader(path, "invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_window_reader_gpu_matches_cpu(tmp_path):
    path = tmp_path / "source.edf"
    make_source(path, annotation_in_middle=True)
    starts = np.array([127, 0, 256, 127])
    cpu = records_module.EdfWindowReader(path).read(starts, 128)
    gpu = records_module.EdfWindowReader(path, "edf_gpu_assembly", torch.device("cuda")).read(starts, 128)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-6, atol=1e-10)


def test_window_reader_pickle_reopens_without_sharing_source_handles(tmp_path):
    import pickle

    path = tmp_path / "source.edf"
    make_source(path)
    reader = records_module.EdfWindowReader(path)
    restored = pickle.loads(pickle.dumps(reader))
    assert restored.records is None and restored.assembler is None
    assert reader.records is not None
    starts = np.array([127, 0, 127])
    torch.testing.assert_close(restored.read(starts, 128), reader.read(starts, 128), rtol=0, atol=0)
    assert restored.records is not reader.records
