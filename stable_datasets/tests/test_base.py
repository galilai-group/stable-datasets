from collections.abc import Mapping

import datasets
import pytest

from stable_datasets.utils import BaseDatasetBuilder


class _TinyLocalBuilder(BaseDatasetBuilder):
    """A tiny local builder used to validate BaseDatasetBuilder return types."""

    VERSION = datasets.Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 3}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"n": 2}),
        ]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class _TinyDynamicSourceBuilder(BaseDatasetBuilder):
    """Validates that BaseDatasetBuilder supports runtime-computed source via _source()."""

    VERSION = datasets.Version("0.0.0")

    def _source(self):
        # In real datasets this might depend on self.config; here it's static but computed at runtime.
        return {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

    def _split_generators(self, dl_manager):
        # Ensure the runtime hook is available and returns an immutable mapping.
        src = self._source()
        assert isinstance(src, Mapping)
        with pytest.raises(TypeError):
            src["homepage"] = "https://mutate.example.com"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 1}),
        ]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class _TinyBaseSplitBuilder(BaseDatasetBuilder):
    """Uses BaseDatasetBuilder's default _split_generators (SOURCE['assets'] + bulk_download)."""

    VERSION = datasets.Version("0.0.0")
    SOURCE = {
        "homepage": "https://example.com",
        "citation": "TBD",
        "assets": {"train": "https://example.com/train.bin", "test": "https://example.com/test.bin"},
    }

    def _info(self):
        return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

    def _generate_examples(self, data_path, split):
        # We don't touch data_path; tests monkeypatch bulk_download to avoid network.
        yield f"{split}-0", {"x": 0}


def test_base_builder_returns_datasetdict_when_split_is_none(tmp_path):
    ds = _TinyLocalBuilder(split=None, processed_cache_dir=str(tmp_path))
    assert isinstance(ds, datasets.DatasetDict)
    assert set(ds.keys()) == {"train", "test"}


def test_base_builder_returns_dataset_when_split_is_set(tmp_path):
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert isinstance(ds, datasets.Dataset)
    assert len(ds) == 3


def test_base_builder_allows_runtime_source_override(tmp_path):
    ds = _TinyDynamicSourceBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert isinstance(ds, datasets.Dataset)
    assert len(ds) == 1


def test_base_builder_processed_cache_dir_is_used(tmp_path):
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    # HuggingFace datasets exposes the underlying Arrow cache files.
    for cache_entry in ds.cache_files:
        assert cache_entry["filename"].startswith(str(tmp_path))

    # stable-datasets also exposes the processed cache location as a convenience attribute.
    assert getattr(ds, "_stable_datasets_processed_cache_dir") == tmp_path


def test_base_builder_passes_download_dir_to_bulk_download(tmp_path, monkeypatch):
    import stable_datasets.utils as utils

    seen = {}

    def _fake_bulk_download(urls, dest_folder, backend="filesystem", cache_dir=utils.DEFAULT_CACHE_DIR):
        seen["dest_folder"] = str(dest_folder)
        # Return fake local paths; _generate_examples ignores data_path.
        return [tmp_path / f"fake_{i}.bin" for i in range(len(list(urls)))]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    download_dir = tmp_path / "raw_downloads"
    _ = _TinyBaseSplitBuilder(
        split="train",
        processed_cache_dir=str(tmp_path / "processed"),
        download_dir=str(download_dir),
    )
    assert seen["dest_folder"] == str(download_dir)


def test_base_builder_requires_version():
    with pytest.raises(TypeError):

        class _MissingVersion(BaseDatasetBuilder):
            SOURCE = {
                "homepage": "https://example.com",
                "citation": "TBD",
                "assets": {"train": "https://example.com/train.bin"},
            }

            def _info(self):
                return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

            def _generate_examples(self, data_path, split):
                yield 0, {"x": 0}


def test_base_builder_requires_source_or_source_override():
    with pytest.raises(TypeError):

        class _MissingSource(BaseDatasetBuilder):
            VERSION = datasets.Version("0.0.0")

            def _info(self):
                return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

            def _generate_examples(self, data_path, split):
                yield 0, {"x": 0}


def test_base_builder_source_override_must_return_mapping(tmp_path):
    class _BadSource(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")

        def _source(self):
            return "not-a-dict"

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _BadSource(split="train", processed_cache_dir=str(tmp_path))


def test_source_is_frozen_for_static_source():
    with pytest.raises(TypeError):
        _TinyLocalBuilder.SOURCE["homepage"] = "https://mutate.example.com"


def test_base_builder_requires_source_homepage(tmp_path):
    class _MissingHomepage(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {"citation": "TBD", "assets": {}}

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _split_generators(self, dl_manager):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 1})]

        def _generate_examples(self, n):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _MissingHomepage(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_requires_source_citation(tmp_path):
    class _MissingCitation(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {"homepage": "https://example.com", "assets": {}}

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _split_generators(self, dl_manager):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 1})]

        def _generate_examples(self, n):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _MissingCitation(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_requires_source_assets(tmp_path):
    class _MissingDownloadUrls(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {"homepage": "https://example.com", "citation": "TBD"}

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _split_generators(self, dl_manager):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 1})]

        def _generate_examples(self, n):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _MissingDownloadUrls(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_validates_source_field_types(tmp_path):
    class _BadTypes(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {"homepage": 123, "citation": object(), "assets": "not-a-dict"}

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _split_generators(self, dl_manager):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"n": 1})]

        def _generate_examples(self, n):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _BadTypes(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_empty_urls_raises(tmp_path):
    class _EmptyUrls(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    with pytest.raises(ValueError):
        _EmptyUrls(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_maps_val_to_validation(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    class _ValSplit(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {
            "homepage": "https://example.com",
            "citation": "TBD",
            "assets": {"val": "https://example.com/all.bin"},
        }

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    def _fake_bulk_download(urls, dest_folder, backend="filesystem", cache_dir=utils.DEFAULT_CACHE_DIR):
        return [tmp_path / "fake.bin" for _ in list(urls)]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    # Bypass __new__/download_and_prepare; directly test split generator mapping.
    inst = object.__new__(_ValSplit)
    inst._raw_download_dir = tmp_path
    splits = inst._split_generators(dl_manager=None)
    assert len(splits) == 1
    assert splits[0].name == datasets.Split.VALIDATION


def test_base_builder_deduplicates_urls(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    class _SharedUrl(BaseDatasetBuilder):
        VERSION = datasets.Version("0.0.0")
        SOURCE = {
            "homepage": "https://example.com",
            "citation": "TBD",
            "assets": {"train": "https://example.com/file.bin", "test": "https://example.com/file.bin"},
        }

        def _info(self):
            return datasets.DatasetInfo(features=datasets.Features({"x": datasets.Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    seen = {}

    def _fake_bulk_download(urls, dest_folder, backend="filesystem", cache_dir=utils.DEFAULT_CACHE_DIR):
        urls = list(urls)
        seen["urls"] = urls
        return [tmp_path / f"fake_{i}.bin" for i in range(len(urls))]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    inst = object.__new__(_SharedUrl)
    inst._raw_download_dir = tmp_path
    _ = inst._split_generators(dl_manager=None)
    assert seen["urls"] == ["https://example.com/file.bin"]
