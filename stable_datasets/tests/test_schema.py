"""Tests for schema containers and metadata helpers."""

from collections import OrderedDict

from stable_datasets.schema import Features, Value


def test_features_is_explicitly_ordered():
    features = Features()
    features["b"] = Value("int32")
    features["a"] = Value("int32")

    assert isinstance(features, OrderedDict)
    assert list(features.keys()) == ["b", "a"]
    assert features.fingerprint_data() == "{'b': Value('int32'), 'a': Value('int32')}"


def test_features_arrow_schema_preserves_insertion_order():
    features = Features({"image": Value("binary"), "label": Value("int32")})

    schema = features.to_arrow_schema()

    assert schema.names == ["image", "label"]


def test_class_label_fingerprint_keeps_every_name():
    from stable_datasets.schema import ClassLabel, Sequence

    names = [f"class_{i}" for i in range(10)]
    renamed = [*names[:-1], "other"]

    assert repr(ClassLabel(names=names)) == repr(ClassLabel(names=renamed))
    assert (
        Features({"y": ClassLabel(names=names)}).fingerprint_data()
        != Features({"y": ClassLabel(names=renamed)}).fingerprint_data()
    )
    assert (
        Features({"y": Sequence(ClassLabel(names=names))}).fingerprint_data()
        != Features({"y": Sequence(ClassLabel(names=renamed))}).fingerprint_data()
    )
