import hashlib
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stable_datasets import utils
from stable_datasets.dataset import StableDataset, StableDatasetDict
from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version, collect_dataset_citations
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder


class _TinyLocalBuilder(BaseDatasetBuilder):
    """A tiny local builder used to validate BaseDatasetBuilder return types."""

    VERSION = Version("0.0.0")
    SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 3}),
            SplitGenerator(name=Split.TEST, gen_kwargs={"n": 2}),
        ]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class _TinyDynamicSourceBuilder(BaseDatasetBuilder):
    """Validates that BaseDatasetBuilder supports runtime-computed source via _source()."""

    VERSION = Version("0.0.0")

    def _source(self):
        # In real datasets this might depend on self.config; here it's static but computed at runtime.
        return {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _split_generators(self):
        # Ensure the runtime hook is available and returns an immutable mapping.
        src = self._source()
        assert isinstance(src, Mapping)
        with pytest.raises(TypeError):
            src["homepage"] = "https://mutate.example.com"
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 1}),
        ]

    def _generate_examples(self, n):
        for i in range(n):
            yield i, {"x": i}


class _TinyBaseSplitBuilder(BaseDatasetBuilder):
    """Uses BaseDatasetBuilder's default _split_generators (SOURCE['assets'] + bulk_download)."""

    VERSION = Version("0.0.0")
    SOURCE = {
        "homepage": "https://example.com",
        "citation": "TBD",
        "assets": {"train": "https://example.com/train.bin", "test": "https://example.com/test.bin"},
    }

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _generate_examples(self, data_path, split):
        # We don't touch data_path; tests monkeypatch bulk_download to avoid network.
        yield f"{split}-0", {"x": 0}


class _TinyFallbackSplitBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = DatasetSource(
        homepage="https://example.com",
        citation="TBD",
        assets={
            "train": DownloadInfo(
                url="https://primary.example.com/file.bin",
                fallbacks=["https://mirror.example.com/file.bin"],
                filename="file.bin",
            ),
            "test": DownloadInfo(
                url="https://primary.example.com/file.bin",
                fallbacks=["https://mirror.example.com/file.bin"],
                filename="file.bin",
            ),
        },
    )

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _generate_examples(self, data_path, split):
        yield f"{split}-0", {"x": 0}


class _TinyMixedAssetBuilder(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = DatasetSource(
        homepage="https://example.com",
        citation="TBD",
        assets={
            "train": "https://example.com/train.bin",
            "test": DownloadInfo(
                url="https://primary.example.com/test.bin",
                fallbacks=["https://mirror.example.com/test.bin"],
            ),
        },
    )

    def _info(self):
        return DatasetInfo(features=Features({"x": Value("int32")}))

    def _generate_examples(self, data_path, split):
        yield f"{split}-0", {"x": 0}


def test_base_builder_returns_datasetdict_when_split_is_none(tmp_path):
    ds = _TinyLocalBuilder(split=None, processed_cache_dir=str(tmp_path))
    assert isinstance(ds, StableDatasetDict)
    assert set(ds.keys()) == {"train", "test"}


def test_base_builder_returns_dataset_when_split_is_set(tmp_path):
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert isinstance(ds, StableDataset)
    assert len(ds) == 3


def test_base_builder_allows_runtime_source_override(tmp_path):
    ds = _TinyDynamicSourceBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert isinstance(ds, StableDataset)
    assert len(ds) == 1


def test_base_builder_processed_cache_dir_is_used(tmp_path):
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    # Verify that sharded cache directories were created in the specified directory.
    shard_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(shard_dirs) > 0

    # stable-datasets also exposes the processed cache location as a convenience attribute.
    assert getattr(ds, "_stable_datasets_processed_cache_dir") == tmp_path


def test_base_builder_passes_download_dir_to_bulk_download(tmp_path, monkeypatch):
    import stable_datasets.utils as utils

    seen = {}

    def _fake_bulk_download(urls, dest_folder, checksums=None):
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


def test_normalize_download_info_from_string():
    info = BaseDatasetBuilder._normalize_download_info("https://example.com/file.bin")
    assert isinstance(info, DownloadInfo)
    assert info.url == "https://example.com/file.bin"
    assert info.fallbacks == []


def test_normalize_download_info_preserves_explicit_downloadinfo():
    explicit = DownloadInfo(url="https://example.com/file.bin", fallbacks=["https://mirror.example.com/file.bin"])
    normalized = BaseDatasetBuilder._normalize_download_info(explicit)
    assert normalized is explicit


def test_normalize_dataset_source_from_dict():
    source = BaseDatasetBuilder._normalize_dataset_source(
        {"homepage": "https://example.com", "citation": "TBD", "assets": {"train": "https://example.com/train.bin"}}
    )
    assert isinstance(source, DatasetSource)
    assert isinstance(source.assets["train"], DownloadInfo)


def test_collect_dataset_citations_deduplicates_in_order():
    citations = collect_dataset_citations(
        [
            DatasetSource(homepage="https://a.example", citation="cite-a", assets={}),
            DatasetSource(homepage="https://b.example", citation="cite-b", assets={}),
            DatasetSource(homepage="https://c.example", citation="cite-a", assets={}),
        ]
    )
    assert citations == ["cite-a", "cite-b"]


def test_base_builder_requires_version():
    with pytest.raises(TypeError):

        class _MissingVersion(BaseDatasetBuilder):
            SOURCE = {
                "homepage": "https://example.com",
                "citation": "TBD",
                "assets": {"train": "https://example.com/train.bin"},
            }

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _generate_examples(self, data_path, split):
                yield 0, {"x": 0}


def test_base_builder_requires_source_or_source_override():
    with pytest.raises(TypeError):

        class _MissingSource(BaseDatasetBuilder):
            VERSION = Version("0.0.0")

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _generate_examples(self, data_path, split):
                yield 0, {"x": 0}


def test_base_builder_source_override_must_return_mapping(tmp_path):
    class _BadSource(BaseDatasetBuilder):
        VERSION = Version("0.0.0")

        def _source(self):
            return "not-a-dict"

        def _info(self):
            return DatasetInfo(features=Features({"x": Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    with pytest.raises(TypeError):
        _BadSource(split="train", processed_cache_dir=str(tmp_path))


def test_source_is_frozen_for_static_source():
    with pytest.raises(TypeError):
        _TinyLocalBuilder.SOURCE["homepage"] = "https://mutate.example.com"


def test_base_builder_requires_source_homepage(tmp_path):
    with pytest.raises(TypeError):
        class _MissingHomepage(BaseDatasetBuilder):
            VERSION = Version("0.0.0")
            SOURCE = {"citation": "TBD", "assets": {}}

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _split_generators(self):
                return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 1})]

            def _generate_examples(self, n):
                yield 0, {"x": 0}


def test_base_builder_requires_source_citation(tmp_path):
    with pytest.raises(TypeError):
        class _MissingCitation(BaseDatasetBuilder):
            VERSION = Version("0.0.0")
            SOURCE = {"homepage": "https://example.com", "assets": {}}

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _split_generators(self):
                return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 1})]

            def _generate_examples(self, n):
                yield 0, {"x": 0}


def test_base_builder_requires_source_assets(tmp_path):
    with pytest.raises(TypeError):
        class _MissingDownloadUrls(BaseDatasetBuilder):
            VERSION = Version("0.0.0")
            SOURCE = {"homepage": "https://example.com", "citation": "TBD"}

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _split_generators(self):
                return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 1})]

            def _generate_examples(self, n):
                yield 0, {"x": 0}


def test_base_builder_validates_source_field_types(tmp_path):
    with pytest.raises(TypeError):
        class _BadTypes(BaseDatasetBuilder):
            VERSION = Version("0.0.0")
            SOURCE = {"homepage": 123, "citation": object(), "assets": "not-a-dict"}

            def _info(self):
                return DatasetInfo(features=Features({"x": Value("int32")}))

            def _split_generators(self):
                return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"n": 1})]

            def _generate_examples(self, n):
                yield 0, {"x": 0}


def test_base_builder_empty_urls_raises(tmp_path):
    class _EmptyUrls(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = {"homepage": "https://example.com", "citation": "TBD", "assets": {}}

        def _info(self):
            return DatasetInfo(features=Features({"x": Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    with pytest.raises(ValueError):
        _EmptyUrls(split="train", processed_cache_dir=str(tmp_path))


def test_base_builder_maps_val_to_validation(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    class _ValSplit(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = {
            "homepage": "https://example.com",
            "citation": "TBD",
            "assets": {"val": "https://example.com/all.bin"},
        }

        def _info(self):
            return DatasetInfo(features=Features({"x": Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        return [tmp_path / "fake.bin" for _ in list(urls)]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    # Bypass __new__; directly test split generator mapping.
    inst = object.__new__(_ValSplit)
    inst._raw_download_dir = tmp_path
    inst.__init__()
    splits = inst._split_generators()
    assert len(splits) == 1
    assert splits[0].name == Split.VALIDATION


def test_base_builder_deduplicates_urls(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    class _SharedUrl(BaseDatasetBuilder):
        VERSION = Version("0.0.0")
        SOURCE = {
            "homepage": "https://example.com",
            "citation": "TBD",
            "assets": {"train": "https://example.com/file.bin", "test": "https://example.com/file.bin"},
        }

        def _info(self):
            return DatasetInfo(features=Features({"x": Value("int32")}))

        def _generate_examples(self, data_path, split):
            yield 0, {"x": 0}

    seen = {}

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        urls = list(urls)
        seen["urls"] = urls
        return [tmp_path / f"fake_{i}.bin" for i in range(len(urls))]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    inst = object.__new__(_SharedUrl)
    inst._raw_download_dir = tmp_path
    inst.__init__()
    _ = inst._split_generators()
    assert len(seen["urls"]) == 1
    assert isinstance(seen["urls"][0], DownloadInfo)
    assert seen["urls"][0].url == "https://example.com/file.bin"


def test_base_builder_deduplicates_downloadinfo_specs(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    seen = {}

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        urls = list(urls)
        seen["urls"] = urls
        return [tmp_path / f"fake_{i}.bin" for i in range(len(urls))]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    inst = object.__new__(_TinyFallbackSplitBuilder)
    inst._raw_download_dir = tmp_path
    inst.__init__()
    _ = inst._split_generators()
    assert len(seen["urls"]) == 1
    assert isinstance(seen["urls"][0], DownloadInfo)
    assert seen["urls"][0].fallbacks == ["https://mirror.example.com/file.bin"]


def test_base_builder_accepts_mixed_asset_specs(monkeypatch, tmp_path):
    import stable_datasets.utils as utils

    seen = {}

    def _fake_bulk_download(urls, dest_folder, checksums=None):
        urls = list(urls)
        seen["urls"] = urls
        return [tmp_path / f"fake_{i}.bin" for i in range(len(urls))]

    monkeypatch.setattr(utils, "bulk_download", _fake_bulk_download)

    inst = object.__new__(_TinyMixedAssetBuilder)
    inst._raw_download_dir = tmp_path
    inst.__init__()
    _ = inst._split_generators()
    assert [type(url).__name__ for url in seen["urls"]] == ["DownloadInfo", "DownloadInfo"]


def test_stable_dataset_getitem(tmp_path):
    """Verify StableDataset supports integer indexing and returns correct values."""
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert ds[0] == {"x": 0}
    assert ds[1] == {"x": 1}
    assert ds[2] == {"x": 2}


def test_stable_dataset_train_test_split(tmp_path):
    """Verify StableDataset.train_test_split produces correct sizes."""
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    splits = ds.train_test_split(test_size=0.34, seed=42)
    assert "train" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["test"]) == 3


def test_stable_dataset_features_property(tmp_path):
    """Verify .features returns the Features dict."""
    ds = _TinyLocalBuilder(split="train", processed_cache_dir=str(tmp_path))
    assert "x" in ds.features
    assert isinstance(ds.features["x"], Value)


# STABLE_DATASETS_CACHE_DIR env variable tests


def test_get_cache_dir_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(utils.CACHE_DIR_ENV_VAR, raising=False)
    assert utils._get_cache_dir() == utils.DEFAULT_CACHE_DIR


def test_get_cache_dir_returns_env_value_when_set(monkeypatch):
    monkeypatch.setenv(utils.CACHE_DIR_ENV_VAR, "/tmp/my_cache")
    assert utils._get_cache_dir() == "/tmp/my_cache"


def test_default_dest_folder_respects_env(monkeypatch):
    custom = "/tmp/custom_stable"
    monkeypatch.setenv(utils.CACHE_DIR_ENV_VAR, custom)
    assert utils._default_dest_folder() == Path(custom) / "downloads"


def test_default_processed_cache_dir_respects_env(monkeypatch):
    custom = "/tmp/custom_stable"
    monkeypatch.setenv(utils.CACHE_DIR_ENV_VAR, custom)
    assert utils._default_processed_cache_dir() == Path(custom) / "processed"


def test_default_dest_folder_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(utils.CACHE_DIR_ENV_VAR, raising=False)
    expected = Path.home() / ".stable-datasets" / "downloads"
    assert utils._default_dest_folder() == expected


def test_default_processed_cache_dir_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(utils.CACHE_DIR_ENV_VAR, raising=False)
    expected = Path.home() / ".stable-datasets" / "processed"
    assert utils._default_processed_cache_dir() == expected


def test_env_cache_dir_with_tilde_expansion(monkeypatch):
    monkeypatch.setenv(utils.CACHE_DIR_ENV_VAR, "~/my_datasets_cache")
    expected = Path.home() / "my_datasets_cache" / "downloads"
    assert utils._default_dest_folder() == expected


def test_download_resume_sends_range_header(tmp_path, monkeypatch):
    """When a .tmp file exists, download should send a Range header."""
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    # Pre-create a partial .tmp file
    url = "https://example.com/file.bin"
    p = Path("file.bin")
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    tmp_file = dest_folder / f"{p.stem}.{h}{p.suffix}.tmp"
    tmp_file.write_bytes(b"partial")

    seen_headers = {}

    class FakeResponse:
        status_code = 200
        headers = {"content-length": "100"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([b"x" * 100])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def update(self, d):
            self.headers.update(d)

        def get(self, url, stream=True, allow_redirects=True, timeout=None, headers=None):
            seen_headers.update(headers or {})
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    FakeSession.headers = MagicMock()
    FakeSession.headers.update = lambda d: None

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            try:
                download(url, dest_folder=dest_folder, progress_bar=False)
            except Exception:
                pass  # We only care about the Range header

    assert "Range" in seen_headers
    assert seen_headers["Range"] == f"bytes={len(b'partial')}-"


def test_download_resume_fallback_on_200(tmp_path, monkeypatch):
    """When server returns 200 (ignoring Range), download starts over."""
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    url = "https://example.com/file2.bin"
    p = Path("file2.bin")
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    dest = dest_folder / f"{p.stem}.{h}{p.suffix}"
    # Ensure no leftover dest
    if dest.exists():
        dest.unlink()

    content = b"fullcontent"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([content])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        headers = {}

        def update(self, d):
            pass

        def get(self, url, **kwargs):
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    FakeSession.headers = MagicMock()
    FakeSession.headers.update = lambda d: None

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            result = download(url, dest_folder=dest_folder, progress_bar=False)

    # File should contain the full content (not appended)
    assert result.read_bytes() == content


def test_download_uses_fallback_url(tmp_path):
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    primary = "https://primary.example.com/file.bin"
    fallback = "https://mirror.example.com/file.bin"
    content = b"fallback-content"
    seen = []

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([content])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            seen.append(url)
            if url == primary:
                raise RuntimeError("primary down")
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            result = download(primary, dest_folder=dest_folder, progress_bar=False, fallbacks=[fallback])

    assert seen == [primary, fallback]
    assert result.read_bytes() == content


def test_download_raises_aggregated_error_when_all_candidates_fail(tmp_path):
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    primary = "https://primary.example.com/file.bin"
    fallback = "https://mirror.example.com/file.bin"

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            raise RuntimeError(f"failed: {url}")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            with pytest.raises(RuntimeError, match="Failed to download from all candidate URLs"):
                download(primary, dest_folder=dest_folder, progress_bar=False, fallbacks=[fallback])


def test_download_validates_checksum_on_fallback(tmp_path):
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    primary = "https://primary.example.com/file.bin"
    fallback = "https://mirror.example.com/file.bin"
    content = b"checksum-on-fallback"
    expected = hashlib.sha256(content).hexdigest()

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([content])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            if url == primary:
                raise RuntimeError("primary down")
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            result = download(
                primary,
                dest_folder=dest_folder,
                progress_bar=False,
                fallbacks=[fallback],
                checksum=f"sha256:{expected}",
            )

    assert result.read_bytes() == content


def test_checksum_validation_passes(tmp_path):
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    content = b"hello world"
    sha = hashlib.sha256(content).hexdigest()
    checksum = f"sha256:{sha}"

    url = "https://example.com/check.bin"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([content])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        headers = {}

        def update(self, d):
            pass

        def get(self, url, **kwargs):
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    FakeSession.headers = MagicMock()
    FakeSession.headers.update = lambda d: None

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            result = download(url, dest_folder=dest_folder, progress_bar=False, checksum=checksum)

    assert result.read_bytes() == content


def test_checksum_mismatch_deletes_and_raises(tmp_path):
    from stable_datasets.utils import download

    dest_folder = tmp_path / "downloads"
    dest_folder.mkdir()

    content = b"hello world"
    wrong_checksum = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    url = "https://example.com/bad.bin"

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            return iter([content])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class FakeSession:
        headers = {}

        def update(self, d):
            pass

        def get(self, url, **kwargs):
            return FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    FakeSession.headers = MagicMock()
    FakeSession.headers.update = lambda d: None

    with patch("stable_datasets.utils.requests.Session", return_value=FakeSession()):
        with patch("stable_datasets.utils.FileLock"):
            with pytest.raises(ValueError, match="Checksum mismatch"):
                download(url, dest_folder=dest_folder, progress_bar=False, checksum=wrong_checksum)
