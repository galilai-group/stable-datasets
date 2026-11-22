"""Check neonatal label alignment and matched learning through real readers."""

import pickle

import numpy as np
import pytest
import torch
from scipy.io import savemat

from profiling.neonatal_targets import load_expert_labels, window_targets


def annotation_file(path, experts):
    cells = np.empty((1, 79), dtype=object)
    for index in range(79):
        cells[0, index] = np.zeros_like(experts)
    cells[0, 3] = experts
    savemat(path, {"annotat_new": cells})


def test_expert_votes_and_window_boundaries_are_explicit(tmp_path):
    experts = np.zeros((3, 6), dtype=np.uint8)
    experts[:2, 1] = 1
    experts[2, 3] = 1
    experts[1:, 5] = 1
    path = tmp_path / "annotations.mat"
    annotation_file(path, experts)
    np.testing.assert_array_equal(load_expert_labels(path, 4), experts)
    np.testing.assert_array_equal(load_expert_labels(path, 3), 0)
    starts = np.array([[0, 8, 16], [16, 0, 8]])
    np.testing.assert_array_equal(window_targets(experts, starts, 8, 4), [[1, 0, 1], [1, 1, 0]])
    for bad in (np.array([1]), np.array([-4]), np.array([20]), np.array([0.0])):
        with pytest.raises(ValueError):
            window_targets(experts, bad, 8, 4)
    with pytest.raises(ValueError):
        window_targets(experts, np.array([0]), 7, 4)
    for subject in (0, 80):
        with pytest.raises(ValueError):
            load_expert_labels(path, subject)
    experts[0, 0] = 2
    annotation_file(path, experts)
    with pytest.raises(ValueError, match="binary"):
        load_expert_labels(path, 4)


@pytest.fixture
def training_case(tmp_path):
    pytest.importorskip("lance")
    pytest.importorskip("mne")
    pytest.importorskip("edfio")
    highlevel = pytest.importorskip("pyedflib.highlevel")
    from profiling.benchmark_signal_windows import prepare

    source = tmp_path / "eeg4.edf"
    t = np.arange(2048)
    samples = np.stack([t - 1000, t // 2 - 700, 1500 - t]).astype(np.int32)
    headers = [
        highlevel.make_signal_header(name, sample_frequency=128, physical_min=-100, physical_max=300)
        for name in ("EEG Fp1-Ref", "EEG Fp2-Ref", "ECG EKG")
    ]
    highlevel.write_edf(str(source), samples, headers, digital=True)
    workspace = tmp_path / "prepared"
    prepare(source, workspace, chunk_samples=200)
    experts = np.zeros((3, 16), dtype=np.uint8)
    experts[:2, [2, 8, 14]] = 1
    annotations = tmp_path / "annotations.mat"
    annotation_file(annotations, experts)
    starts = np.array([[0, 256], [1024, 1792], [256, 256], [0, 1536]])
    targets = window_targets(experts, starts, 256, 128)
    return source, workspace, annotations, starts, targets


@pytest.mark.parametrize("path", ["current_cache", "edf_memmap", "lance_blob_ranges", "lance_chunk_rows"])
def test_cpu_batch_readers_match_mne_and_pickle_without_open_handles(training_case, path):
    import mne

    from profiling.edf_reference import read_mne_windows
    from profiling.training_windows import WindowBatches

    source, workspace, _, starts, targets = training_case
    dataset = WindowBatches(source, workspace, starts, targets, 256, path)
    with mne.io.read_raw_edf(str(source), preload=False, verbose="ERROR") as raw:
        for index in range(len(dataset)):
            actual = dataset[index]
            reference = read_mne_windows(raw, starts[index], 256)
            np.testing.assert_allclose(actual["values"], reference, rtol=2e-6, atol=1e-10)
            np.testing.assert_array_equal(actual["targets"], targets[index])
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._reader is None
    np.testing.assert_array_equal(restored[1]["values"], dataset[1]["values"])


@pytest.mark.parametrize("path", ["current_cache", "edf_memmap", "lance_blob_ranges", "lance_chunk_rows"])
def test_spawned_readers_preserve_order_and_learning(training_case, path):
    from profiling.benchmark_signal_training import train_trial
    from profiling.training_windows import WindowBatches, configure_reader_threads

    configure_reader_threads()
    source, workspace, _, starts, targets = training_case
    reference = train_trial(
        WindowBatches(source, workspace, starts, targets, 256, "edf_memmap"), 0, torch.device("cpu")
    )
    result = train_trial(WindowBatches(source, workspace, starts, targets, 256, path), 2, torch.device("cpu"))
    np.testing.assert_allclose(
        [step["loss"] for step in result["steps"]],
        [step["loss"] for step in reference["steps"]],
        rtol=2e-4,
        atol=2e-5,
    )
    assert result["iterator_startup_seconds"] > 0
    assert result["elapsed_seconds"] >= sum(step["data_ready_seconds"] for step in result["steps"])


def test_model_excludes_auxiliary_channels_from_forward_and_backward():
    from profiling.benchmark_signal_training import SmallEegCnn

    model = SmallEegCnn(("EEG Fp1-Ref", "EEG Fp2-Ref", "ECG EKG"))
    values = (torch.randn(2, 256, 3) * 1e-4).requires_grad_()
    changed = values.detach().clone()
    changed[:, :, 2] += 100
    torch.testing.assert_close(model(values), model(changed), rtol=0, atol=0)
    torch.nn.functional.cross_entropy(model(values), torch.tensor([0, 1])).backward()
    assert torch.count_nonzero(values.grad[:, :, 2]) == 0
    assert torch.count_nonzero(values.grad[:, :, :2]) > 0


def test_diagnostic_validates_matched_training_and_rejects_wrong_patient(training_case):
    from profiling.benchmark_signal_training import run

    source, workspace, annotations, _, _ = training_case
    args = {"subject_id": 4, "batch_size": 2, "steps": 3, "seconds": 2, "workers": [0], "trials": 2, "device": "cpu"}
    result = run(source, workspace, annotations, **args)
    assert len(result["trials"]) == 8
    assert result["parameters"] > 0
    assert len(result["targets"]) == 3
    assert max(result["first_batch_max_abs_errors_volts"].values()) < 1e-10
    assert all(len(trial["steps"]) == 3 for trial in result["trials"])
    with pytest.raises(ValueError, match="subject ID"):
        run(source, workspace, annotations, **{**args, "subject_id": 1})


def test_existing_gpu_path_cannot_be_moved_into_workers(training_case):
    from profiling.benchmark_signal_training import train_trial
    from profiling.training_windows import GPU_PATH, WindowBatches

    source, workspace, _, starts, targets = training_case
    dataset = WindowBatches(source, workspace, starts, targets, 256, GPU_PATH)
    assert set(dataset[0]) == {"starts", "targets"}
    assert dataset._reader is None
    with pytest.raises(ValueError, match="zero workers"):
        train_trial(dataset, 2, torch.device("cuda"))
