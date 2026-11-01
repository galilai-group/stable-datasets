"""Offline tests for the FSDnoisy18k SimCLR example (``examples/fsd_simclr.py``)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "fsd_simclr", Path(__file__).resolve().parents[3] / "examples" / "fsd_simclr.py"
)
example = importlib.util.module_from_spec(_SPEC)
sys.modules["fsd_simclr"] = example
_SPEC.loader.exec_module(example)


def write_features(root: Path, subset: str, clips, labels=None, names=("a", "b", "c")):
    """Store ``clips`` (list of ``[T, 96]`` arrays) in the example's feature layout."""
    root.mkdir(parents=True, exist_ok=True)
    labels = [i % len(names) for i in range(len(clips))] if labels is None else labels
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in clips])])
    meta = [
        {"filename": f"{subset}{i}.wav", "label": int(label), "frames": len(c), "offset": int(o)}
        for i, (c, label, o) in enumerate(zip(clips, labels, offsets))
    ]
    np.save(root / f"{subset}.npy", np.concatenate(clips).astype(np.float32))
    (root / f"{subset}.json").write_text(json.dumps({"label_names": list(names), "clips": meta}))
    return example.Features(root, subset)


# -- Front end ---------------------------------------------------------------


def test_mel_filterbank_band_limits_and_order():
    filters = example.mel_filterbank()
    freqs = np.linspace(0, example.SAMPLE_RATE / 2, example.N_FFT // 2 + 1)
    assert filters.shape == (96, 1025)
    assert np.all(filters[:, (freqs < example.F_MIN) | (freqs > example.F_MAX)] == 0)
    assert np.all(np.diff(filters.argmax(axis=1)) >= 0)
    assert filters.max() <= 1


def test_log_mel_matches_reference_pipeline():
    """Same features as uclser20 ``get_mel_spectrogram`` (librosa STFT + Slaney filters, no area norm)."""
    librosa = pytest.importorskip("librosa")
    from scipy.signal import get_window

    rng = np.random.default_rng(0)
    wave = rng.standard_normal(22050 * 2) * np.sin(np.linspace(0, 30, 22050 * 2))
    y = wave - wave.mean()
    y = y / (np.abs(y).max() + 0.005)
    power = (
        np.abs(
            librosa.stft(
                y + example.LOG_EPS,
                n_fft=2048,
                hop_length=220,
                win_length=660,
                window=get_window("hamming", 660, fftbins=True),
                center=True,
                pad_mode="reflect",
            )
        )
        ** 2
    )
    filters = librosa.filters.mel(sr=22050, n_fft=2048, n_mels=96, fmin=50, fmax=10500, htk=False, norm=None)
    expected = np.log10(filters @ power + example.LOG_EPS).T
    np.testing.assert_allclose(example.mel_filterbank(), filters, atol=1e-7)
    np.testing.assert_allclose(example.LogMel()(wave), expected, atol=1e-5)


def test_log_mel_frames_replication_and_gain_invariance():
    rng = np.random.default_rng(1)
    front_end = example.LogMel()
    short = rng.standard_normal(8000)
    assert front_end(short).shape == (example.PATCH_LEN, 96)  # replicated up to 1 s
    long = rng.standard_normal(int(2.5 * 22050))
    features = front_end(long)
    assert features.shape == (1 + len(long) // example.HOP_LENGTH, 96)
    np.testing.assert_allclose(front_end(long + 3), features, atol=1e-4)  # mean removal
    # Peak normalisation with a fixed 0.005 headroom: a gain change only shifts the log energies.
    shift = front_end(1000 * long) - features
    np.testing.assert_allclose(shift, shift.mean(), atol=1e-4)
    with pytest.raises(ValueError):
        front_end(np.zeros(0))


@pytest.mark.parametrize("frames", [101, 150, 151, 250, 1001])
def test_patch_starts_match_reference(frames):
    expected, start = [], 0
    while start + example.PATCH_LEN <= frames:  # uclser20 ``fetch_file_2_tensor``
        expected.append(start)
        start += example.PATCH_HOP
    assert example.patch_starts(frames) == expected


def test_build_features_assigns_subsets_from_builder_columns(tmp_path):
    rng = np.random.default_rng(2)
    rows = {
        "train": [(0.5, 1, True), (2.0, 0, False), (1.2, 2, False)],
        "test": [(1.0, 2, True)],
    }

    class FakeBuilder:
        def __init__(self, split, config_name):
            assert config_name == "22khz"
            items = rows[split]
            series = [rng.standard_normal((int(s * 22050), 1)).astype(np.float32).tolist() for s, _, _ in items]
            self.table = pa.table(
                {
                    "series": pa.array(series, type=pa.list_(pa.list_(pa.float32()))),
                    "sample_rate": [22050] * len(items),
                    "label": [label for _, label, _ in items],
                    "filename": [f"{split}{i}.wav" for i in range(len(items))],
                    "manually_verified": [clean for _, _, clean in items],
                }
            )
            self.features = {"label": type("Label", (), {"names": ["x", "y", "z"]})}

    counts = example.build_features(tmp_path, builder=FakeBuilder)
    assert counts == {"noisy": 2, "clean": 1, "test": 1}
    noisy = example.Features(tmp_path, "noisy")
    assert noisy.labels.tolist() == [0, 2]
    assert noisy.filenames == ["train1.wav", "train2.wav"]
    assert noisy.lengths.tolist() == [1 + 2 * 22050 // 220, 1 + int(1.2 * 22050) // 220]
    assert noisy.offsets.tolist() == [0, noisy.lengths[0]]
    assert noisy.frames.shape == (noisy.lengths.sum(), 96)
    assert example.Features(tmp_path, "clean").lengths.tolist() == [example.PATCH_LEN]
    assert example.Features(tmp_path, "test").label_names == ["x", "y", "z"]


def test_build_features_rejects_wrong_sample_rate(tmp_path):
    class FakeBuilder:
        def __init__(self, split, config_name):
            self.table = pa.table(
                {
                    "series": pa.array([[[0.1], [0.2]]], type=pa.list_(pa.list_(pa.float32()))),
                    "sample_rate": [44100],
                    "label": [0],
                    "filename": ["a.wav"],
                    "manually_verified": [False],
                }
            )
            self.features = {"label": type("Label", (), {"names": ["x"]})}

    with pytest.raises(ValueError, match="22050"):
        example.build_features(tmp_path, builder=FakeBuilder)


def test_features_pickle_without_copying_frames(tmp_path):
    import pickle

    features = write_features(tmp_path, "noisy", [np.ones((101, 96)), np.zeros((120, 96))])
    payload = pickle.dumps(features)
    assert len(payload) < features.frames.nbytes
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.clip(1), features.clip(1))


def test_normalization_fits_training_subset_only(tmp_path):
    noisy = write_features(tmp_path, "noisy", [np.zeros((101, 96)), np.full((101, 96), 2.0)])
    write_features(tmp_path, "clean", [np.full((101, 96), 100.0)])
    mean, std = example.normalization_stats(noisy, "standard")
    assert (mean, std) == pytest.approx((1.0, 1.0))
    mean, std = example.normalization_stats(noisy, "reference")
    assert std == pytest.approx(1.0 / np.sqrt(2 * 101 * 96))  # uclser20: sqrt(sum of squares) / N
    with pytest.raises(ValueError):
        example.normalization_stats(noisy, "other")


# -- Augmentations -----------------------------------------------------------


def test_mix_back_keeps_patch_dominant_with_alpha_energy():
    rng = np.random.default_rng(3)
    patch = rng.uniform(-6, 0, (101, 96)).astype(np.float32)
    background = rng.uniform(-2, 2, (101, 96)).astype(np.float32)  # much louder than the patch
    np.testing.assert_allclose(example.mix_back(patch, background, 0.0), patch, atol=1e-5)
    alpha = 0.05
    mixed = 10.0 ** example.mix_back(patch, background, alpha).astype(np.float64)
    added = mixed - (1 - alpha) * 10.0 ** patch.astype(np.float64)
    assert added.sum() == pytest.approx(alpha * (10.0 ** patch.astype(np.float64)).sum(), rel=1e-4)
    share = 10.0**background / (10.0**background).sum()
    np.testing.assert_allclose(added / added.sum(), share, rtol=0, atol=1e-3 * share.max())


def test_random_resized_crop_shape_and_identity():
    rng = np.random.default_rng(4)
    spec = torch.randn(101, 96)
    for _ in range(20):
        assert example.random_resized_crop(spec, rng).shape == (101, 96)
    full = example.random_resized_crop(spec, rng, scale=(1, 1), ratio=(96 / 101, 96 / 101))
    torch.testing.assert_close(full, spec)


def test_simclr_views_use_two_distinct_out_of_clip_backgrounds(tmp_path, monkeypatch):
    clips = [np.full((150 + 10 * i, 96), float(i)) for i in range(4)]
    features = write_features(tmp_path, "noisy", clips)
    seen = []

    def record(patch, background, alpha):
        assert 0 <= alpha <= 0.05
        seen.append((patch[0, 0], background[0, 0]))
        return patch

    monkeypatch.setattr(example, "mix_back", record)
    torch.manual_seed(0)
    dataset = example.SimCLRPatches(features, mean=0.0, std=1.0)
    for index in range(len(dataset)):
        seen.clear()
        item = dataset[index]
        assert set(item) == {"views"}  # pretraining never sees labels
        assert [set(v) for v in item["views"]] == [{"image"}, {"image"}]
        assert all(v["image"].shape == (1, 101, 96) for v in item["views"])
        (anchor_a, back_a), (anchor_b, back_b) = seen
        assert anchor_a == anchor_b == index
        assert back_a != back_b and index not in (back_a, back_b)


def test_simclr_views_differ_and_leave_features_untouched(tmp_path):
    rng = np.random.default_rng(5)
    features = write_features(tmp_path, "noisy", [rng.uniform(-5, 0, (300, 96)) for _ in range(3)])
    before = np.array(features.frames)
    torch.manual_seed(0)
    first, second = example.SimCLRPatches(features, mean=-2.5, std=1.0)[0]["views"]
    assert not torch.equal(first["image"], second["image"])
    np.testing.assert_array_equal(features.frames, before)


# -- Loss, pretraining and probe ------------------------------------------------


def _uclser20_loss(z_i, z_j, temperature):
    """Loss of uclser20 ``train_Contrastive`` (MIT, Eduardo Fonseca), copied for an equivalence check."""
    bsz = z_i.shape[0]
    zs = torch.cat([z_i, z_j], dim=0)
    logits = torch.div(torch.matmul(zs, zs.t()), temperature)
    mask = torch.eye(bsz, dtype=torch.float32).repeat(2, 2)
    logits_mask = torch.ones_like(mask) - torch.eye(2 * bsz)
    mask = mask * logits_mask
    exp_logits_den = torch.log((torch.exp(logits) * logits_mask).sum(1, keepdim=True) + 1e-10)
    log_prob = torch.log(torch.exp(logits) + 1e-10) - exp_logits_den
    return (-(mask * log_prob).sum(1) / mask.sum(1)).view(2, bsz).mean()


def test_stable_pretraining_ntxent_matches_reference_loss():
    pytest.importorskip("stable_pretraining")
    from stable_pretraining.losses import NTXEntLoss

    torch.manual_seed(0)
    z_i, z_j = torch.randn(16, 512), torch.randn(16, 512)
    reference = _uclser20_loss(
        torch.nn.functional.normalize(z_i, dim=1), torch.nn.functional.normalize(z_j, dim=1), 0.2
    )
    torch.testing.assert_close(NTXEntLoss(temperature=0.2)(z_i, z_j), reference)


def test_pretraining_step_is_seeded_and_updates_the_encoder(tmp_path):
    pytest.importorskip("stable_pretraining")
    rng = np.random.default_rng(6)
    features = write_features(tmp_path / "f", "noisy", [rng.uniform(-6, 0, (120, 96)) for _ in range(8)])
    initial = example.make_encoder().state_dict()
    encoders = []
    for run in range(2):
        out = tmp_path / f"run{run}"
        out.mkdir()
        history = example.pretrain(
            features, -3.0, 1.0, out, seed=0, epochs=1, batch_size=4, workers=0, precision="32-true"
        )
        assert len(history) == 1 and np.isfinite(history[0]["loss"])
        encoders.append(torch.load(out / "encoder.pt", weights_only=True))
    torch.testing.assert_close(encoders[0], encoders[1])
    assert not torch.equal(encoders[0]["layer1.0.conv1.weight"], initial["layer1.0.conv1.weight"])


def test_pretraining_decays_learning_rate_at_epoch_milestones(tmp_path):
    """The paper divides the lr by 10 at epoch milestones, not after that many optimizer steps."""
    pytest.importorskip("stable_pretraining")
    rng = np.random.default_rng(9)
    features = write_features(tmp_path / "f", "noisy", [rng.uniform(-6, 0, (120, 96)) for _ in range(8)])
    out = tmp_path / "run"
    out.mkdir()
    history = example.pretrain(
        features, -3.0, 1.0, out, seed=0, epochs=3, batch_size=4, workers=0, precision="32-true", milestones=(2,)
    )
    # Two steps per epoch; the lr recorded after each epoch drops once epoch 2 has finished.
    assert [row["lr"] for row in history] == pytest.approx([0.03, 0.003, 0.003])


def test_clip_predictions_average_patch_probabilities():
    logits = torch.tensor([[5.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 3.0]])
    clip_ids = torch.tensor([0, 0, 0, 1])
    # Clip 0: one confident patch for class 0 outweighs two weak ones for class 1.
    assert example.clip_predictions(logits, clip_ids, 2).tolist() == [0, 1]


def test_balanced_accuracy_averages_recall_over_present_classes():
    labels = np.array([0, 0, 0, 1])
    predictions = np.array([0, 0, 0, 0])
    assert example.balanced_accuracy(labels, predictions, n_classes=3) == pytest.approx(0.5)


def _toy_split(rng, n_clips, n_classes=3, patches=3):
    labels = np.arange(n_clips) % n_classes
    centers = np.eye(n_classes, 8) * 3
    embeddings = torch.tensor(
        np.concatenate([centers[label] + rng.normal(0, 1, (patches, 8)) for label in labels]), dtype=torch.float32
    )
    return embeddings, torch.arange(n_clips).repeat_interleave(patches), labels


def test_linear_probe_selects_on_validation_and_never_uses_test_labels():
    rng = np.random.default_rng(7)
    train, val, test = _toy_split(rng, 60), _toy_split(rng, 15), _toy_split(rng, 15)
    first, history, prediction, _ = example.linear_probe(train, val, test, n_classes=3, seed=0, epochs=12)
    shuffled = (test[0], test[1], rng.permutation(test[2]))
    second, history2, prediction2, _ = example.linear_probe(train, val, shuffled, n_classes=3, seed=0, epochs=12)
    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["validation"] == second["validation"] and history == history2
    np.testing.assert_array_equal(prediction, prediction2)
    best = max(history, key=lambda row: row["balanced_accuracy"])
    assert first["validation"]["balanced_accuracy"] == best["balanced_accuracy"]
    assert first["test"]["balanced_accuracy"] > 0.9


def test_load_encoder_accepts_reference_checkpoint_layout(tmp_path):
    reference = example.make_encoder()
    reference.fc = torch.nn.Linear(512, 20)  # uclser20 checkpoints keep their 20-way layer
    torch.save(reference.state_dict(), tmp_path / "ResNet_best.pth")
    encoder = example.load_encoder(str(tmp_path / "ResNet_best.pth"), seed=0)
    torch.testing.assert_close(encoder.conv1.weight, reference.conv1.weight)
    assert encoder(torch.randn(2, 1, 101, 96)).shape == (2, 512)
    torch.testing.assert_close(
        example.load_encoder("random", 3).state_dict(), example.load_encoder("random", 3).state_dict()
    )


def test_probe_end_to_end_writes_metrics(tmp_path):
    rng = np.random.default_rng(8)
    root = tmp_path / "features"
    for subset, n in (("noisy", 9), ("clean", 3), ("test", 3)):
        write_features(root, subset, [rng.uniform(-6, 0, (rng.integers(101, 260), 96)) for _ in range(n)])
    result = example.probe(root, "random", tmp_path / "probe", 0, "standard", "cpu", bn_batches=2)
    assert set(result["test"]) == {"balanced_accuracy", "accuracy"}
    saved = json.loads((tmp_path / "probe" / "metrics.json").read_text())
    assert saved["selected_epoch"] == result["selected_epoch"]
    assert np.load(tmp_path / "probe" / "predictions.npz")["prediction"].shape == (3,)
