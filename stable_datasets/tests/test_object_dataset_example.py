"""Regression checks for leakage prevention in the small SSL example."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import torch
from PIL import Image

from stable_datasets.dataset import StableDataset
from stable_datasets.schema import ClassLabel, DatasetInfo, Features, Value


spec = importlib.util.spec_from_file_location(
    "object_example", Path(__file__).resolve().parents[2] / "examples/object_datasets_lejepa.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_validation_split_groups_duplicate_images():
    ids = [f"class{label}/image{i}.png" for label in range(2) for i in range(20)]
    labels = [label for label in range(2) for _ in range(20)]
    features = Features({"image_id": Value("string"), "label": ClassLabel(names=["class0", "class1"])})
    data = StableDataset(
        table=pa.table({"image_id": ids + ids, "label": labels + labels}),
        features=features,
        info=DatasetInfo(features=features),
    )
    train, val = example.partition_freiburg(data, 0.2, 42)
    train_ids = {row["image_id"] for row in train}
    val_ids = {row["image_id"] for row in val}
    assert len(train) == 32 and len(val) == 8
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(ids)
    assert {row["label"] for row in train} == {0, 1}
    assert {row["label"] for row in val} == {0, 1}
    again = example.partition_freiburg(data, 0.2, 42)
    assert all(list(a) == list(b) for a, b in zip((train, val), again))


def test_ssl_views_do_not_expose_labels():
    views = example.Views({"image_size": 64, "local_size": 32})
    sample = views({"image": Image.new("RGB", (256, 256)), "label": 13, "image_id": "test.png"})
    assert set(sample) == {"global_views", "local_views"}
    assert [tuple(v.shape) for v in sample["global_views"]] == [(3, 64, 64)] * 2
    assert [tuple(v.shape) for v in sample["local_views"]] == [(3, 32, 32)] * 2


def test_probe_scaling_uses_training_features_only():
    x = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    config = {"seed": 42, "probe_epochs": 3, "probe_weight_decays": [0.0]}
    _, mean, std, chosen, _ = example.fit_probe((x, y), (x + 100, y), 2, config)
    assert torch.equal(mean, x.mean(0))
    assert torch.equal(std, x.std(0).clamp_min(1e-6))
    assert 1 <= chosen["epoch"] <= 3


def test_extracted_frozen_features_can_train_a_probe():
    import io

    from stable_datasets.schema import Image as ImageFeature

    encoded = []
    for value in (0, 30, 200, 255):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (value, value, value)).save(buffer, format="PNG")
        encoded.append(buffer.getvalue())
    features = Features({"image": ImageFeature(), "label": ClassLabel(names=["dark", "light"])})
    data = StableDataset(
        table=pa.table({"image": encoded, "label": [0, 0, 1, 1]}),
        features=features,
        info=DatasetInfo(features=features),
    )
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4))
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    config = {
        "image_size": 8,
        "batch_size": 4,
        "num_workers": 0,
        "seed": 42,
        "probe_epochs": 3,
        "probe_weight_decays": [0.0],
    }
    extracted = example.embeddings(encoder, data, config, "cpu")
    head, _, _, _, _ = example.fit_probe(extracted, extracted, 2, config)
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    assert all(torch.isfinite(p).all() for p in head.parameters())
