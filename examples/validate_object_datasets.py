"""Opt-in real-data validation of ETH80 and FreiburgGroceries (downloads required)."""

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from stable_datasets.images import ETH80, FreiburgGroceries
from stable_datasets.utils import download


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate(builder, cache_root):
    kwargs = {"download_dir": cache_root / "downloads", "processed_cache_dir": cache_root / "processed"}
    ds = builder(**kwargs)
    source_key = "train" if builder is ETH80 else "images"
    paths = {key: download(spec, dest_folder=kwargs["download_dir"]) for key, spec in builder.SOURCE.assets.items()}
    checksums = {key: digest(path) for key, path in paths.items()}
    for key, checksum in checksums.items():
        assert builder.SOURCE.assets[key].checksum == f"sha256:{checksum}"
    reference = {}
    with tarfile.open(paths[source_key], "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".png") or "/maps/" in member.name:
                continue
            with archive.extractfile(member) as file, Image.open(io.BytesIO(file.read())) as image:
                rgb = image.convert("RGB")
                assert rgb.size == (256, 256)
                image_id = Path(member.name).name if builder is ETH80 else member.name.removeprefix("images/")
                assert image_id not in reference
                reference[image_id] = hashlib.sha256(rgb.tobytes()).hexdigest()
    assert len(reference) == (3280 if builder is ETH80 else 4947)
    summary, ids, objects = {}, {}, {}
    for split, data in ds.items():
        ids[split], objects[split] = set(), set()
        counts = Counter()
        for row in data:
            assert row["image"].mode == "RGB" and row["image"].size == (256, 256)
            assert hashlib.sha256(row["image"].tobytes()).hexdigest() == reference[row["image_id"]]
            ids[split].add(row["image_id"])
            counts[row["label"]] += 1
            if builder is ETH80:
                objects[split].add(row["object_id"])
                assert row["object_id"] == f"{builder.LABELS[row['label']]}{row['instance_id']}"
            else:
                assert row["image_id"].split("/")[0] == builder.LABELS[row["label"]]
        summary[split] = {"rows": len(data), "unique_images": len(ids[split]), "class_counts": dict(counts)}
    for a, b in combinations(ds, 2):
        assert not ids[a] & ids[b]
        assert not objects[a] & objects[b]
    assert set.union(*ids.values()) == set(reference)
    if builder is ETH80:
        assert {k: len(v) for k, v in ds.items()} == {"train": 1968, "validation": 656, "test": 656}
        assert {k: len(v) for k, v in objects.items()} == {"train": 48, "validation": 16, "test": 16}
    else:
        prefix = f"freiburg_groceries_dataset-{builder.SPLITS_REVISION}"
        folds = {}
        with tarfile.open(paths["annotations"], "r:*") as archive:
            for fold in range(5):
                partitions = {}
                for split in ("train", "test"):
                    text = archive.extractfile(f"{prefix}/splits/{split}{fold}.txt").read().decode()
                    rows = [(line.split()[0], int(line.split()[1])) for line in text.splitlines()]
                    assert all(
                        path in reference and path.split("/")[0] == builder.LABELS[label] for path, label in rows
                    )
                    partitions[split] = {path for path, _ in rows}
                    if fold == 0:
                        actual = ds[split].remove_columns("image")
                        assert [(r["image_id"], r["label"]) for r in actual] == rows
                assert not partitions["train"] & partitions["test"]
                folds[str(fold)] = {name: len(values) for name, values in partitions.items()}
        summary["author_folds_unique_images"] = folds
    reopened = builder(**kwargs)
    for split in ds:
        assert ds[split].table.equals(reopened[split].table)
        batch = next(iter(DataLoader(reopened[split].with_format("torch"), batch_size=8)))
        assert tuple(batch["image"].shape) == (8, 3, 256, 256)
    return {
        "source_checksums": checksums,
        "source_image_count": len(reference),
        "splits": summary,
        "all_pixels_match_source": True,
        "cache_reopen_equal": True,
        "batch_shape": [8, 3, 256, 256],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {builder.__name__: validate(builder, args.cache_root) for builder in (ETH80, FreiburgGroceries)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
