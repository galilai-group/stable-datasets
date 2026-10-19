"""Protocol invariants: stable assignment, complete coverage and group isolation."""

import json
from collections import defaultdict

import pytest
from PIL import Image

from benchmarks.protocols import ImageRecord, fingerprint_samples, isic2016_protocol, scene8_protocol


def _records(classes=3, per_class=25, prefix="image"):
    return [
        ImageRecord(f"{prefix}_{label}_{i}", label, f"{prefix}_pixels_{label}_{i}")
        for label in range(classes)
        for i in range(per_class)
    ]


def _assert_partition(manifest, source, *, test_group_disjoint=True):
    roles = {role: set(ids) for role, ids in manifest.roles.items()}
    assert not roles["fit"] & roles["validation"]
    assert not roles["trainval"] & roles["test"]
    assert roles["fit"] | roles["validation"] == roles["trainval"]
    assert roles["trainval"] | roles["test"] | set(manifest.excluded_train_ids) == {r.image_id for r in source}
    by_role = {name: {r.pixel_sha256 for r in source if r.image_id in ids} for name, ids in roles.items()}
    assert not by_role["fit"] & by_role["validation"]
    if test_group_disjoint:
        assert not by_role["trainval"] & by_role["test"]


def test_scene8_groups_and_conflicting_labels_stay_together():
    records = _records()
    records += [
        ImageRecord("same_class_duplicate", 0, records[0].pixel_sha256),
        ImageRecord("conflicting_duplicate", 1, records[1].pixel_sha256),
    ]
    protocol = scene8_protocol(records)
    _assert_partition(protocol, records)
    assert len(protocol.duplicate_groups) == 2
    assert protocol.conflicting_label_groups == [["conflicting_duplicate", records[1].image_id]]
    counts = defaultdict(set)
    for role in ("fit", "validation", "test"):
        for record in records:
            if record.image_id in protocol.roles[role]:
                counts[record.pixel_sha256].add(role)
    assert all(len(roles) == 1 for roles in counts.values())


def test_protocol_is_input_order_invariant_and_seeded(tmp_path):
    records = _records()
    first = scene8_protocol(records, seed=42)
    assert first == scene8_protocol(list(reversed(records)), seed=42)
    assert first.roles != scene8_protocol(records, seed=43).roles
    assert first.source_fingerprint == scene8_protocol(records, seed=43).source_fingerprint
    path = tmp_path / "manifest.json"
    first.save(path)
    saved = json.loads(path.read_text())
    assert saved["roles"] == first.roles
    assert saved["source_fingerprint"] == first.source_fingerprint
    changed = records[:-1] + [ImageRecord(records[-1].image_id, records[-1].label, "changed_pixels")]
    assert first.source_fingerprint != scene8_protocol(changed).source_fingerprint


def test_isic_official_and_filtered_keep_test_unchanged():
    train = _records(classes=2, prefix="train")
    test = _records(classes=2, per_class=6, prefix="test")
    test[0] = ImageRecord(test[0].image_id, 0, train[0].pixel_sha256)
    train += [ImageRecord("train_duplicate", 0, train[0].pixel_sha256)]
    official = isic2016_protocol(train, test)
    filtered = isic2016_protocol(train, test, exclude_test_duplicates=True)
    _assert_partition(official, train + test, test_group_disjoint=False)
    _assert_partition(filtered, train + test)
    assert official.roles["test"] == filtered.roles["test"] == sorted(r.image_id for r in test)
    assert official.excluded_train_ids == []
    assert filtered.excluded_train_ids == ["train_0_0", "train_duplicate"]
    assert official.source_fingerprint == filtered.source_fingerprint
    assert official.roles["validation"] == filtered.roles["validation"]
    for role in ("fit", "trainval"):
        assert set(official.roles[role]) - set(filtered.excluded_train_ids) == set(filtered.roles[role])
    assert not set(official.roles["validation"]) & set(filtered.excluded_train_ids)
    assert official.protocol.endswith("v2") and filtered.protocol.endswith("v2")


def test_isic_assignments_do_not_depend_on_test_diagnoses():
    train = _records(classes=2, prefix="train")
    test = _records(classes=2, per_class=6, prefix="test")
    test[0] = ImageRecord(test[0].image_id, 0, train[0].pixel_sha256)
    changed_test = [ImageRecord(r.image_id, 1 - r.label, r.pixel_sha256) for r in test]
    for filtered in (False, True):
        first = isic2016_protocol(train, test, exclude_test_duplicates=filtered)
        second = isic2016_protocol(train, changed_test, exclude_test_duplicates=filtered)
        assert first.roles == second.roles
        assert first.excluded_train_ids == second.excluded_train_ids
        # Provenance still records the labels; only assignments ignore them.
        assert first.source_fingerprint != second.source_fingerprint


@pytest.mark.parametrize(
    "records, message",
    [
        ([], "empty"),
        (_records()[:2], "five distinct"),
        ([ImageRecord("a", 0, "x"), ImageRecord("a", 1, "y")], "unique"),
        ([ImageRecord("a", -1, "x")], "nonnegative"),
    ],
)
def test_protocol_rejects_invalid_or_insufficient_sources(records, message):
    with pytest.raises(ValueError, match=message):
        scene8_protocol(records)


def test_isic_rejects_source_id_overlap():
    records = _records(classes=2)
    with pytest.raises(ValueError, match="IDs overlap"):
        isic2016_protocol(records, records)


def test_fingerprint_includes_dimensions_and_preserves_identity():
    image = Image.new("RGB", (4, 4), "red")
    rows = [
        {"image_id": "a", "label": 0, "image": image},
        {"image_id": "b", "label": 1, "image": image.copy()},
        {"image_id": "c", "label": 0, "image": Image.new("RGB", (2, 8), "red")},
    ]
    first, second, third = fingerprint_samples(rows)
    assert first.image_id == "a" and second.label == 1
    assert first.pixel_sha256 == second.pixel_sha256
    assert first.pixel_sha256 != third.pixel_sha256


def test_fingerprint_uses_bounded_indexed_reads_for_dataset_objects():
    class IndexedOnly:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"image_id": str(index), "label": index % 2, "image": Image.new("RGB", (3, 3), "red")}

        def __iter__(self):
            raise AssertionError("Do not bulk-decode high-resolution dataset batches.")

    assert [r.image_id for r in fingerprint_samples(IndexedOnly())] == ["0", "1", "2"]
