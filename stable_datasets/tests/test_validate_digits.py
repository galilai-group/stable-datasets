"""Lightweight tests for the supervised digit validation example."""

import io
import wave
import zipfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from examples.validate_digits import (
    TinyAudioCNN,
    TinyImageCNN,
    TrainConfig,
    _AudioAdapter,
    _ImageAdapter,
    audio_collate,
    fsdd_split_indices,
    stratified_split_indices,
    train_and_evaluate,
)


def test_stratified_split_is_reproducible_disjoint_and_complete():
    labels = np.concatenate([np.full(19 + label, label) for label in range(10)])

    first = stratified_split_indices(labels, seed=42)
    second = stratified_split_indices(labels, seed=42)
    different_seed = stratified_split_indices(labels, seed=7)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert any(not np.array_equal(first[name], different_seed[name]) for name in first)

    split_sets = {name: set(indices.tolist()) for name, indices in first.items()}
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])
    assert set.union(*split_sets.values()) == set(range(len(labels)))

    for label in range(10):
        total = int((labels == label).sum())
        assert abs(int((labels[first["train"]] == label).sum()) - 0.8 * total) <= 1
        assert abs(int((labels[first["validation"]] == label).sum()) - 0.1 * total) <= 1
        assert abs(int((labels[first["test"]] == label).sum()) - 0.1 * total) <= 1


def test_fsdd_split_keeps_official_test_and_validation_boundaries():
    official_train_ids = np.tile(np.arange(5, 50), 3)
    official_test_ids = np.tile(np.arange(0, 5), 3)
    splits = fsdd_split_indices(official_train_ids, official_test_ids)

    assert set(official_train_ids[splits["train"]]) == set(range(10, 50))
    assert set(official_train_ids[splits["validation"]]) == set(range(5, 10))
    assert np.array_equal(splits["test"], np.arange(len(official_test_ids)))

    with pytest.raises(ValueError, match="official train"):
        fsdd_split_indices([4, 10], official_test_ids)
    with pytest.raises(ValueError, match="official test"):
        fsdd_split_indices(official_train_ids, [0, 5])


def test_audio_collate_pads_to_bct_and_preserves_lengths():
    batch = audio_collate(
        [
            {"series": np.arange(5, dtype=np.float32)[:, None], "label": 2},
            {"series": np.arange(8, dtype=np.float32)[:, None], "label": 7},
        ]
    )

    assert batch["series"].shape == (2, 1, 8)
    assert batch["lengths"].tolist() == [5, 8]
    assert batch["label"].tolist() == [2, 7]
    assert torch.equal(batch["series"][0, 0, :5], torch.arange(5, dtype=torch.float32))
    assert torch.count_nonzero(batch["series"][0, 0, 5:]) == 0


def test_audio_forward_backward_is_finite_and_updates_parameters():
    torch.manual_seed(0)
    batch = audio_collate(
        [
            {"series": torch.randn(129, 1), "label": 1},
            {"series": torch.randn(173, 1), "label": 4},
            {"series": torch.randn(211, 1), "label": 9},
        ]
    )
    model = TinyAudioCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    logits = model(batch["series"], batch["lengths"])
    loss = F.cross_entropy(logits, batch["label"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert logits.shape == (3, 10)
    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_audio_masked_pooling_is_independent_of_batch_padding():
    torch.manual_seed(1)
    model = TinyAudioCNN().eval()
    short = torch.randn(127, 1)
    long = torch.randn(301, 1)

    alone = audio_collate([{"series": short, "label": 0}])
    together = audio_collate([{"series": short, "label": 0}, {"series": long, "label": 1}])
    with torch.inference_mode():
        alone_logits = model(alone["series"], alone["lengths"])[0]
        together_logits = model(together["series"], together["lengths"])[0]

    torch.testing.assert_close(alone_logits, together_logits, atol=1e-6, rtol=1e-5)


def test_image_forward_backward_is_finite_and_updates_parameters():
    torch.manual_seed(2)
    model = TinyImageCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    images = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 4, 9])
    before = [parameter.detach().clone() for parameter in model.parameters()]

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert logits.shape == (3, 10)
    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 10), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _wav_bytes(length: int) -> bytes:
    samples = (np.sin(np.linspace(0, 8 * np.pi, length)) * 12000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())
    return buffer.getvalue()


def _single_update(model, logits, labels):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_builder_derived_batches_update_both_models(tmp_path, monkeypatch):
    from stable_datasets.images import SignLanguageDigits
    from stable_datasets.timeseries import FSDD

    sign_archive = tmp_path / "sign.zip"
    with zipfile.ZipFile(sign_archive, "w") as archive:
        archive.writestr("snapshot/Dataset/0/a.png", _png_bytes((10, 20, 30)))
        archive.writestr("snapshot/Dataset/9/b.png", _png_bytes((200, 180, 160)))

    def sign_download(urls, dest_folder, checksums=None):
        del urls, dest_folder, checksums
        return [sign_archive]

    monkeypatch.setattr("stable_datasets.utils.bulk_download", sign_download)
    sign_dataset = SignLanguageDigits(
        split="all",
        processed_cache_dir=tmp_path / "sign-processed",
        download_dir=tmp_path / "sign-downloads",
    )
    image_batch = next(iter(DataLoader(_ImageAdapter(sign_dataset, 16), batch_size=2)))
    image_model = TinyImageCNN()
    _single_update(image_model, image_model(image_batch["image"]), image_batch["label"])

    fsdd_archive = tmp_path / "fsdd.zip"
    with zipfile.ZipFile(fsdd_archive, "w") as archive:
        archive.writestr("snapshot/recordings/0_george_0.wav", _wav_bytes(83))
        archive.writestr("snapshot/recordings/1_jackson_5.wav", _wav_bytes(127))
        archive.writestr("snapshot/recordings/2_theo_10.wav", _wav_bytes(191))

    def fsdd_download(urls, dest_folder, checksums=None):
        del urls, dest_folder, checksums
        return [fsdd_archive]

    monkeypatch.setattr("stable_datasets.timeseries.fsdd.bulk_download", fsdd_download)
    fsdd_dataset = FSDD(
        split="train",
        processed_cache_dir=tmp_path / "fsdd-processed",
        download_dir=tmp_path / "fsdd-downloads",
    )
    audio_batch = next(iter(DataLoader(_AudioAdapter(fsdd_dataset), batch_size=2, collate_fn=audio_collate)))
    assert audio_batch["series"].shape == (2, 1, 191)
    assert audio_batch["lengths"].tolist() == [127, 191]
    audio_model = TinyAudioCNN()
    audio_logits = audio_model(audio_batch["series"], audio_batch["lengths"])
    _single_update(audio_model, audio_logits, audio_batch["label"])


def test_tiny_stable_pretraining_manager_run_writes_artifacts(tmp_path):
    pytest.importorskip("stable_pretraining")
    pytest.importorskip("lightning")

    class TensorImages(Dataset):
        def __init__(self, count: int):
            generator = torch.Generator().manual_seed(11 + count)
            self.images = torch.randn(count, 3, 16, 16, generator=generator)
            self.labels = torch.arange(count) % 2

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {"image": self.images[index], "label": self.labels[index]}

    loaders = {
        "train": DataLoader(TensorImages(6), batch_size=2, shuffle=False),
        "validation": DataLoader(TensorImages(4), batch_size=2, shuffle=False),
        "test": DataLoader(TensorImages(4), batch_size=2, shuffle=False),
    }
    output_dir = tmp_path / "spt-run"
    results = train_and_evaluate(
        model=TinyImageCNN(num_classes=2),
        loaders=loaders,
        output_dir=output_dir,
        train_config=TrainConfig(
            dataset="tiny-test",
            seed=42,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_workers=0,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            smoke_test=True,
        ),
        experiment_config={"fixture": True},
    )

    assert results["engine"]["runner"] == "stable_pretraining.Manager"
    assert results["history"][0]["train"]["samples"] == 4
    assert results["history"][0]["validation"]["samples"] == 2
    assert results["test"]["samples"] == 2
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / results["artifacts"]["checkpoint"]).is_file()
