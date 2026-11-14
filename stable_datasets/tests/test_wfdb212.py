"""Reference-based checks for 12-bit signs, missing samples and native windows."""

import importlib

import numpy as np
import pytest
import torch


wfdb = pytest.importorskip("wfdb")
numba = pytest.importorskip("numba")
source_module = importlib.import_module("profiling.wfdb212")
cpu = importlib.import_module("profiling.wfdb212_cpu")


@pytest.fixture
def record(tmp_path):
    """Cover every signed 12-bit value, random pairs, two units and nonzero baselines."""
    values = np.arange(-2048, 2048, dtype=np.int16)
    random = np.random.default_rng(0).integers(-2048, 2048, size=(4096, 2), dtype=np.int16)
    digital = np.concatenate((np.column_stack((values, values[::-1])), random))
    wfdb.wrsamp(
        "known",
        fs=360,
        units=["mV", "uV"],
        sig_name=["first", "second"],
        d_signal=digital,
        fmt=["212", "212"],
        adc_gain=[200, 400],
        baseline=[17, -31],
        write_dir=str(tmp_path),
    )
    return tmp_path / "known", digital


def test_unpack_all_signed_values_and_random_pairs_against_reference(record):
    path, digital = record
    source = source_module.Wfdb212Source(path)
    starts = np.array([0])
    packed = source.gather(starts, len(digital))
    np.testing.assert_array_equal(source_module.unpack_cpu(packed)[0], digital)
    np.testing.assert_array_equal(source.reference(starts, len(digital), physical=False)[0], digital)
    expected = source.reference(starts, len(digital), physical=True)
    actual = source.physical(source_module.unpack_cpu(packed))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("parallel", [False, True])
def test_fused_cpu_baselines_keep_missing_values_and_units(record, parallel):
    path, digital = record
    source = source_module.Wfdb212Source(path)
    packed = source.gather(np.array([0]), len(digital))
    decoder = cpu.decode_cpu_parallel if parallel else cpu.decode_cpu_serial
    if parallel:
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    actual = decoder(packed, source.scale, source.baseline)
    expected = source.reference(np.array([0]), len(digital), physical=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10, equal_nan=True)


def test_source_byte_offset_odd_windows_duplicates_and_end_boundary(record):
    path, digital = record
    data_path = path.with_suffix(".dat")
    data_path.write_bytes(b"0123456789abcdef" + data_path.read_bytes())
    header = path.with_suffix(".hea")
    header.write_text(header.read_text().replace(" 212 ", " 212+16 "))
    source = source_module.Wfdb212Source(path)
    starts = np.array([1, len(digital) - 257, 1, 0])
    actual = source_module.unpack_cpu(source.gather(starts, 257))
    np.testing.assert_array_equal(actual, np.stack([digital[start : start + 257] for start in starts]))
    np.testing.assert_array_equal(actual, source.reference(starts, 257, physical=False))


@pytest.mark.parametrize(
    "starts,length",
    [
        (np.array([-1]), 1),
        (np.array([8192]), 1),
        (np.array([], dtype=np.int64), 1),
        (np.array([0.1]), 1),
        (np.array([0]), 0),
    ],
)
def test_invalid_windows_fail(record, starts, length):
    source = source_module.Wfdb212Source(record[0])
    with pytest.raises(ValueError):
        source.gather(starts, length)


def test_unsupported_header_and_truncated_data_fail(record):
    path, _ = record
    header = path.with_suffix(".hea")
    original = header.read_text()
    header.write_text(original.replace(" 212 ", " 212:1 "))
    with pytest.raises(ValueError, match="skew"):
        source_module.Wfdb212Source(path)
    header.write_text(original.replace(" 212 ", " 16 "))
    with pytest.raises(ValueError, match="two-channel"):
        source_module.Wfdb212Source(path)
    header.write_text(original)
    data_path = path.with_suffix(".dat")
    data_path.write_bytes(data_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        source_module.Wfdb212Source(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_fused_cuda_is_bit_exact_and_preserves_nan_semantics(record):
    pytest.importorskip("triton")
    from profiling.wfdb212_cuda import calibrate_cuda, decode_cuda

    path, digital = record
    source = source_module.Wfdb212Source(path)
    packed = source.gather(np.array([0]), len(digital))
    raw = torch.from_numpy(packed).cuda()
    scale, baseline = torch.from_numpy(source.scale).cuda(), torch.from_numpy(source.baseline).cuda()
    np.testing.assert_array_equal(decode_cuda(raw, scale, baseline, physical=False).cpu().numpy()[0], digital)
    expected = source.reference(np.array([0]), len(digital), physical=True)
    for actual in (
        decode_cuda(raw, scale, baseline),
        calibrate_cuda(torch.from_numpy(digital[None]).cuda(), scale, baseline),
    ):
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-6, atol=1e-10, equal_nan=True)
    with pytest.raises(ValueError, match="uint8"):
        decode_cuda(raw.float(), scale, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_complete_paths_match_reference_and_preserve_original_files(record):
    pytest.importorskip("triton")
    from profiling.benchmark_wfdb212 import run_benchmark

    path, _ = record
    before = path.with_suffix(".dat").read_bytes()
    report = run_benchmark(path, 3, 17, rounds=1, repeats=1)
    assert report["checks"]["digital_bit_exact"]
    assert report["checks"]["physical_values_and_nan_masks"]
    assert report["source_unchanged"]
    assert len(report["results"]) == 6
    assert path.with_suffix(".dat").read_bytes() == before
    whole = run_benchmark(path, 3, 8192, rounds=1, repeats=1)
    assert whole["window_starts"] == [0, 0, 0]
    assert whole["checks"]["digital_bit_exact"]
