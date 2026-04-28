import hashlib
import multiprocessing
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import rich.progress
from filelock import FileLock
from loguru import logger as logging
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

from .cache import (
    cache_fingerprint,
    detect_cache_format,
    open_cache,
    write_lance_cache,
    write_lance_video_frames_cache,
    write_sharded_arrow_cache,
)
from .dataset import StableDataset, StableDatasetDict
from .schema import BuilderConfig, DatasetSource, DownloadInfo, Image, Version, Video
from .splits import Split, SplitGenerator


DEFAULT_CACHE_DIR = "~/.stable-datasets/"

CACHE_DIR_ENV_VAR = "STABLE_DATASETS_CACHE_DIR"


def _get_cache_dir() -> str:
    """Return the base cache directory, respecting the environment variable."""
    return os.environ.get(CACHE_DIR_ENV_VAR, DEFAULT_CACHE_DIR)


def _default_dest_folder() -> Path:
    """Default folder where files are saved."""
    return Path(os.path.expanduser(_get_cache_dir())) / "downloads"


def _default_processed_cache_dir() -> Path:
    """Default folder where processed datasets (Arrow files) are cached."""
    return Path(os.path.expanduser(_get_cache_dir())) / "processed"


class BaseDatasetBuilder:
    """
    Base class for stable-datasets builders.

    Handles downloading, Arrow caching, and split generation. Subclasses
    implement ``_info``, ``_split_generators``, and ``_generate_examples``.
    """

    # Subclasses must define:
    # - VERSION: schema.Version
    #
    # For dataset provenance / downloads, subclasses can either:
    # - define a class attribute SOURCE (static), or
    # - override _source(self) to compute it at runtime (e.g. from self.config)
    VERSION: Version
    SOURCE: DatasetSource | Mapping

    # Optional: multi-variant datasets define BUILDER_CONFIGS and optionally DEFAULT_CONFIG_NAME.
    BUILDER_CONFIGS: list = []
    DEFAULT_CONFIG_NAME: str | None = None

    # Class-level default storage backend for the processed cache.
    # "arrow" (library default) writes sharded Arrow IPC files via
    # write_sharded_arrow_cache and reads through :class:`ArrowBackend`.
    # "lance" writes a native Lance dataset via write_lance_cache and
    # reads through :class:`LanceBackend`.
    #
    # End users can also override this per-call at load time by
    # passing ``storage_format="arrow"`` or ``storage_format="lance"``
    # to the builder constructor, e.g. ``CIFAR10(split="train",
    # storage_format="lance")``. Arrow and Lance caches coexist at
    # distinct paths because ``cache_fingerprint`` hashes the format,
    # so flipping the format at load time does not invalidate or
    # clobber the other cache.
    #
    # Decision guide (from our profiling on ImageNet-1K and the small
    # image datasets): keep "arrow" for datasets under ~100 GB on
    # 256 GB training nodes -- Arrow's mmap is unbeatable when the
    # dataset fits warm in the page cache. Use "lance" for datasets
    # that will thrash the page cache (above ~300 GB, or any size on
    # sub-128 GB nodes), and for cheap-decode workloads (tensors,
    # audio, embeddings) where decode does not mask the storage-layer
    # difference. See benchmarks/results/lance_investigation_table.tex
    # for the supporting measurements.
    STORAGE_FORMAT: str = "arrow"

    @staticmethod
    def _freeze(obj):
        """
        Recursively freeze basic Python containers to make SOURCE effectively immutable.

        - dict / Mapping -> MappingProxyType(dict(...)) (shallowly immutable mapping)
        - list / tuple -> tuple(...)
        - set -> frozenset(...)
        """
        if isinstance(obj, MappingProxyType):
            return obj
        if isinstance(obj, DatasetSource):
            return obj
        if isinstance(obj, DownloadInfo):
            return DownloadInfo(
                url=obj.url,
                fallbacks=list(obj.fallbacks),
                checksum=obj.checksum,
                filename=obj.filename,
            )
        if isinstance(obj, Mapping):
            # Create a fresh dict so callers can't retain a handle to the mutable original.
            return MappingProxyType({k: BaseDatasetBuilder._freeze(v) for k, v in dict(obj).items()})
        if isinstance(obj, list | tuple):
            return tuple(BaseDatasetBuilder._freeze(v) for v in obj)
        if isinstance(obj, set):
            return frozenset(BaseDatasetBuilder._freeze(v) for v in obj)
        return obj

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Don't validate the base class itself
        if cls is BaseDatasetBuilder:
            return

        # Allow tests / internal helpers to opt out if needed
        if getattr(cls, "_SKIP_SOURCE_VALIDATION", False):
            return

        # VERSION must exist and be a schema.Version
        if not hasattr(cls, "VERSION"):
            raise TypeError(f"{cls.__name__} must define a class attribute VERSION = Version('x.y.z').")
        if not isinstance(getattr(cls, "VERSION"), Version):
            raise TypeError(f"{cls.__name__}.VERSION must be a Version instance.")

        # Enforce that a source is provided either statically (SOURCE) or dynamically (_source override).
        has_static_source = hasattr(cls, "SOURCE")
        has_dynamic_source = cls._source is not BaseDatasetBuilder._source  # overridden
        if not (has_static_source or has_dynamic_source):
            raise TypeError(
                f"{cls.__name__} must define SOURCE = {{...}} or override _source(self) to compute it at runtime."
            )

        # Freeze static SOURCE at class creation time.
        if has_static_source:
            cls.SOURCE = cls._normalize_dataset_source(getattr(cls, "SOURCE"))

        # If subclass overrides _source(), wrap it so the returned mapping is frozen/immutable.
        if has_dynamic_source and not getattr(cls._source, "_stable_datasets_freezes_source", False):
            original = cls._source

            def _wrapped_source(self):
                source = original(self)
                return BaseDatasetBuilder._normalize_dataset_source(source)

            _wrapped_source._stable_datasets_freezes_source = True  # type: ignore[attr-defined]
            cls._source = _wrapped_source  # type: ignore[method-assign]

    def __init__(self, config_name: str | None = None, **kwargs):
        """Initialize builder, selecting a BuilderConfig if applicable."""
        if self.BUILDER_CONFIGS:
            if config_name is None:
                config_name = self.DEFAULT_CONFIG_NAME or self.BUILDER_CONFIGS[0].name
            matched = [c for c in self.BUILDER_CONFIGS if c.name == config_name]
            if not matched:
                available = [c.name for c in self.BUILDER_CONFIGS]
                raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
            self.config = matched[0]
        else:
            self.config = BuilderConfig(name=config_name or "default")

        # Populate self.info / self._dataset_info so _generate_examples can use it
        self._dataset_info = self._info()
        self._dataset_info.config_name = self.config.name

    @property
    def info(self):
        return self._dataset_info

    def _source(self) -> DatasetSource:
        """
        Return dataset provenance / download configuration.

        Default: uses a class attribute SOURCE (frozen into an immutable Mapping).
        Override in subclasses when the source depends on runtime config (e.g. self.config.variant).
        """
        if not hasattr(self.__class__, "SOURCE"):
            raise TypeError(f"{self.__class__.__name__} does not define SOURCE and did not override _source().")
        return self._normalize_dataset_source(getattr(self.__class__, "SOURCE"))

    @staticmethod
    def _normalize_dataset_source(source) -> DatasetSource:
        """Normalize a builder source object to DatasetSource."""
        if isinstance(source, DatasetSource):
            return source
        if isinstance(source, Mapping):
            homepage = source.get("homepage")
            citation = source.get("citation")
            license = source.get("license", "")
            assets = source.get("assets")
            checksums = source.get("checksums")
            return DatasetSource(
                homepage=homepage,
                assets=assets,
                citation=citation,
                license=license,
                checksums=checksums,
            )
        raise TypeError(f"SOURCE must be a DatasetSource or mapping, got {type(source).__name__}.")

    @staticmethod
    def _validate_source(source: DatasetSource | Mapping) -> None:
        source = BaseDatasetBuilder._normalize_dataset_source(source)
        for asset_name, spec in source["assets"].items():
            BaseDatasetBuilder._normalize_download_info(spec, asset_name=asset_name)

    @staticmethod
    def _normalize_download_info(asset_spec, *, asset_name: str | None = None) -> DownloadInfo:
        """Normalize a source asset spec to a DownloadInfo instance."""
        if isinstance(asset_spec, str):
            return DownloadInfo(url=asset_spec)
        if isinstance(asset_spec, DownloadInfo):
            return asset_spec
        if asset_name is None:
            raise TypeError("Asset spec must be a URL string or DownloadInfo.")
        raise TypeError(
            f"SOURCE['assets']['{asset_name}'] must be a URL string or DownloadInfo, "
            f"got {type(asset_spec).__name__}."
        )

    def _split_generators(self):
        """
        Default split generator implementation.

        Most stable-datasets follow the pattern "one downloadable file per split", expressed
        via `SOURCE["assets"]`. Datasets with different layouts can override this method.
        """
        source = self._source()
        self._validate_source(source)

        assets = source["assets"]
        if len(assets) == 0:
            raise ValueError(f"{self.__class__.__name__}.SOURCE['assets'] is empty; cannot infer splits.")

        split_names = list(assets.keys())
        download_infos = {
            split_name: self._normalize_download_info(assets[split_name], asset_name=split_name)
            for split_name in split_names
        }

        # Build primary URL -> checksum mapping from SOURCE["checksums"] if present
        checksums_by_split = source.get("checksums") or {}
        download_specs = []
        for split_name in split_names:
            cs = checksums_by_split.get(split_name) if checksums_by_split else None
            info = download_infos[split_name]
            if cs and info.checksum is None:
                info = DownloadInfo(
                    url=info.url,
                    fallbacks=list(info.fallbacks),
                    checksum=cs,
                    filename=info.filename,
                )
            download_specs.append(info)

        # Deduplicate assets by their full candidate URL set so shared mirrors only download once.
        unique_specs = list(
            dict.fromkeys(
                (info.url, tuple(info.fallbacks), info.checksum, info.filename) for info in download_specs
            )
        )
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        unique_paths = bulk_download(
            [
                DownloadInfo(
                    url=primary,
                    fallbacks=list(fallbacks),
                    checksum=checksum,
                    filename=filename,
                )
                for primary, fallbacks, checksum, filename in unique_specs
            ],
            dest_folder=download_dir,
        )
        spec_to_path = dict(zip(unique_specs, unique_paths))
        local_paths = [
            spec_to_path[(info.url, tuple(info.fallbacks), info.checksum, info.filename)] for info in download_specs
        ]

        split_to_path = dict(zip(split_names, local_paths))

        name_map = {
            "train": Split.TRAIN,
            "test": Split.TEST,
            "val": Split.VALIDATION,
        }

        return [
            SplitGenerator(
                name=name_map.get(split_name, split_name),
                gen_kwargs={"data_path": split_to_path[split_name], "split": split_name},
            )
            for split_name in split_names
        ]

    def _info(self):
        """Return a DatasetInfo describing this dataset's schema. Must be overridden."""
        raise NotImplementedError

    def _generate_examples(self, **kwargs):
        """Yield (key, example_dict) pairs. Must be overridden."""
        raise NotImplementedError

    def _candidate_splits(self) -> list:
        """Return split names used for cache lookup before any download occurs."""
        source = self._source()
        _name_map = {
            "train": Split.TRAIN,
            "test": Split.TEST,
            "val": Split.VALIDATION,
        }
        asset_split_names = list(source["assets"].keys())
        return [_name_map.get(s, s) for s in asset_split_names]

    def __new__(
        cls,
        *args,
        split=None,
        processed_cache_dir=None,
        download_dir=None,
        storage_format=None,
        backend_kwargs=None,
        decode_video=None,
        **kwargs,
    ):
        """
        Automatically download, prepare, and return the dataset for the specified split.

        Args:
            split: Dataset split to load (e.g., "train", "test", "validation"). If None,
                loads all available splits and returns a StableDatasetDict.
            processed_cache_dir: Cache directory for processed datasets. If None,
                defaults to ~/.stable_datasets/processed/.
            download_dir: Directory for raw downloads (ZIP/NPZ/etc). If None, defaults to
                ~/.stable_datasets/downloads/.
            storage_format: Load-time override for the storage backend, either
                ``"arrow"`` or ``"lance"``. When None (default), falls back to
                the builder class's ``STORAGE_FORMAT`` attribute, which itself
                defaults to ``"arrow"``. Arrow and Lance caches coexist at
                distinct directories (``cache_fingerprint`` hashes the
                format), so switching at load time does not invalidate or
                clobber the other cache; the first load in a new format
                pays a cache-miss build, subsequent loads hit instantly.
                Note this is distinct from :meth:`StableDataset.with_format`,
                which controls the output type (torch / numpy / raw) of
                decoded rows.
            decode_video: Optional read-time video decode config or mapping.
                This is applied after cache open/build and is not part of the
                cache fingerprint.
            **kwargs: Additional arguments passed to the dataset builder (e.g. config_name).

        Returns:
            Union[StableDataset, StableDatasetDict]: The loaded dataset (single split)
                or a StableDatasetDict (all splits).
        """
        if storage_format is not None and storage_format not in ("arrow", "lance"):
            raise ValueError(
                f"storage_format must be 'arrow' or 'lance', got {storage_format!r}"
            )
        backend_kwargs = dict(backend_kwargs or {})

        instance = object.__new__(cls)
        effective_format = storage_format if storage_format is not None else cls.STORAGE_FORMAT
        instance._effective_storage_format = effective_format

        # 1) Decide cache locations
        if processed_cache_dir is None:
            processed_cache_dir = str(_default_processed_cache_dir())
        instance._processed_cache_dir = Path(processed_cache_dir)

        if download_dir is None:
            download_dir = str(_default_dest_folder())
        instance._raw_download_dir = Path(download_dir)

        # 2) Initialize builder (handles config selection, calls _info())
        instance.__init__(*args, **kwargs)

        # 3) Validate dataset SOURCE contract early.
        source = instance._source()
        cls._validate_source(source)

        features = instance._dataset_info.features
        has_frame_video = any(isinstance(f, Video) and f.storage == "frames" for f in features.values())
        if has_frame_video and effective_format != "lance":
            raise ValueError(
                "Video(storage='frames') requires storage_format='lance' because "
                "it uses the specialized lance-video-frames layout."
            )

        # 4) Try to load all splits from cache without downloading.
        candidate_splits = instance._candidate_splits()

        splits_data = {}
        all_cached = bool(candidate_splits)  # empty assets → must call _split_generators
        for split_name in candidate_splits:
            shard_dir_name = cache_fingerprint(
                cls.__name__,
                str(cls.VERSION),
                instance.config.name,
                split_name,
                storage_format=effective_format,
            )
            shard_dir = instance._processed_cache_dir / shard_dir_name

            if (shard_dir / "_metadata.json").exists():
                on_disk_format = detect_cache_format(shard_dir)
                if on_disk_format != effective_format:
                    # Different format lives at a different fingerprint;
                    # if we got here the hash collided, which would be a
                    # cache-fingerprint bug. Treat as a cache miss.
                    all_cached = False
                    break
                opened = open_cache(shard_dir, features, backend_kwargs=backend_kwargs)
                splits_data[split_name] = StableDataset(
                    features=features,
                    info=instance._dataset_info,
                    backend=opened.backend,
                    num_rows=opened.num_rows,
                )
            else:
                all_cached = False
                break

        # 5) Cache miss: download raw files and generate caches.
        #    Writer and backend are chosen by effective_format.
        if not all_cached:
            splits_data = {}
            split_generators = instance._split_generators()

            for sg in split_generators:
                shard_dir_name = cache_fingerprint(
                    cls.__name__,
                    str(cls.VERSION),
                    instance.config.name,
                    sg.name,
                    storage_format=effective_format,
                )
                shard_dir = instance._processed_cache_dir / shard_dir_name

                if has_frame_video:
                    if not (shard_dir / "_metadata.json").exists():
                        generator = instance._generate_examples(**sg.gen_kwargs)
                        write_lance_video_frames_cache(
                            generator,
                            features,
                            shard_dir,
                        )
                    opened = open_cache(shard_dir, features, backend_kwargs=backend_kwargs)
                    splits_data[sg.name] = StableDataset(
                        features=features,
                        info=instance._dataset_info,
                        backend=opened.backend,
                        num_rows=opened.num_rows,
                    )
                elif effective_format == "lance":
                    if not (shard_dir / "_metadata.json").exists():
                        generator = instance._generate_examples(**sg.gen_kwargs)
                        write_lance_cache(
                            generator, features, shard_dir, num_encode_workers=4
                        )
                    opened = open_cache(shard_dir, features, backend_kwargs=backend_kwargs)
                    splits_data[sg.name] = StableDataset(
                        features=features,
                        info=instance._dataset_info,
                        backend=opened.backend,
                        num_rows=opened.num_rows,
                    )
                else:
                    if not (shard_dir / "_metadata.json").exists():
                        generator = instance._generate_examples(**sg.gen_kwargs)
                        has_images = any(isinstance(f, Image) for f in features.values())
                        compression = None if has_images else "zstd"
                        write_sharded_arrow_cache(
                            generator, features, shard_dir, compression=compression, num_encode_workers=4
                        )
                    opened = open_cache(shard_dir, features, backend_kwargs=backend_kwargs)
                    splits_data[sg.name] = StableDataset(
                        features=features,
                        info=instance._dataset_info,
                        backend=opened.backend,
                        num_rows=opened.num_rows,
                    )

        # 6) Return single split or dict
        if split is not None:
            if split not in splits_data:
                available = list(splits_data.keys())
                raise ValueError(f"Split '{split}' not found. Available: {available}")
            result = splits_data[split]
        else:
            result = StableDatasetDict(splits_data)

        if decode_video is not None:
            result = result.set_video_decode(decode_video)

        # Expose cache locations on the returned dataset object for convenience.
        try:
            setattr(result, "_stable_datasets_processed_cache_dir", instance._processed_cache_dir)
        except Exception:
            pass
        try:
            setattr(result, "_stable_datasets_download_dir", instance._raw_download_dir)
        except Exception:
            pass
        return result


def bulk_download(
    urls: Iterable[str | DownloadInfo],
    dest_folder: str | Path,
    checksums: dict[str, str] | None = None,
) -> list[Path]:
    """
    Download multiple files concurrently and return their local paths.

    Args:
        urls: Iterable of URL strings or DownloadInfo specs to download.
        dest_folder: Destination folder for downloads.
        checksums: Optional dict mapping primary URL -> checksum string.

    Returns:
        list[Path]: Local file paths in the same order as the input URLs.
    """
    checksums = checksums or {}
    download_infos = []
    for asset_spec in urls:
        info = BaseDatasetBuilder._normalize_download_info(asset_spec)
        if info.checksum is None and checksums.get(info.url):
            info = DownloadInfo(
                url=info.url,
                fallbacks=list(info.fallbacks),
                checksum=checksums[info.url],
                filename=info.filename,
            )
        download_infos.append(info)
    num_workers = min(len(download_infos), os.cpu_count() or 4, 8)
    if num_workers == 0:
        return []

    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    filenames = [
        info.filename or os.path.basename(urlparse(info.url).path) or f"download_{idx}"
        for idx, info in enumerate(download_infos)
    ]
    results: list[Path] = [None] * len(download_infos)

    with rich.progress.Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        refresh_per_second=5,
    ) as progress:
        futures = []
        with multiprocessing.Manager() as manager:
            _progress = manager.dict()  # shared between worker processes

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # submit one download task per URL
                for i in range(len(download_infos)):
                    task_id = f"{i}:{filenames[i]}"
                    info = download_infos[i]
                    future = executor.submit(
                        download,
                        info.url,
                        dest_folder,
                        False,  # disable per-file tqdm; Rich handles progress
                        _progress,
                        task_id,
                        info.checksum,
                        info.fallbacks,
                        info.filename,
                        info.url,
                    )
                    futures.append((i, future))

                rich_tasks = {}

                # update Rich progress while downloads are running
                while not all(future.done() for _, future in futures):
                    for task_id, prog in list(_progress.items()):
                        if task_id not in rich_tasks:
                            rich_tasks[task_id] = progress.add_task(
                                f"[green]{task_id}",
                                total=prog["total"],
                                visible=True,
                            )
                        progress.update(
                            rich_tasks[task_id],
                            completed=prog["progress"],
                        )
                    time.sleep(0.01)

            # collect results in the same order as urls
            for i, future in futures:
                results[i] = future.result()

    return results


def download(
    url: str | DownloadInfo,
    dest_folder: str | Path | None = None,
    progress_bar: bool = True,
    _progress_dict=None,
    _task_id=None,
    checksum: str | None = None,
    fallbacks: list[str] | None = None,
    filename: str | None = None,
    cache_key_url: str | None = None,
) -> Path:
    """Download a file to *dest_folder*, returning the local path.

    Supports resumable downloads: if a ``.tmp`` file exists from a
    previous interrupted attempt, an HTTP ``Range`` header is sent.
    The server may respond with 206 (append) or 200 (start over).

    *checksum*, when provided, is an ``"algorithm:hex"`` string (e.g.
    ``"sha256:a3f8..."``).  The downloaded file is verified after
    completion and deleted on mismatch.
    """
    info = BaseDatasetBuilder._normalize_download_info(url)
    if checksum is None:
        checksum = info.checksum
    if fallbacks is None:
        fallbacks = list(info.fallbacks)
    if filename is None:
        filename = info.filename
    if cache_key_url is None:
        cache_key_url = info.url
    url = info.url

    if dest_folder is None:
        dest_folder = _default_dest_folder()
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    source_url = cache_key_url or url
    candidate_urls = [url, *(fallbacks or [])]
    resolved_filename = filename or os.path.basename(urlparse(source_url).path)
    p = Path(resolved_filename)
    h = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:10]
    # Keep original extension at the end: e.g. cars196_test.0275d128da.zip
    dest = dest_folder / f"{p.stem}.{h}{p.suffix}"
    lock = dest.with_suffix(dest.suffix + ".lock")
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    with FileLock(lock):
        # If you trust your pipeline never leaves a partial 'dest', this is OK.
        # Otherwise, prefer a size/checksum validation here.
        if dest.exists():
            return dest

        try:
            errors = []
            total_size = 0
            for candidate_url in candidate_urls:
                try:
                    with requests.Session() as session:
                        session.headers.update({"User-Agent": "stable-datasets"})

                        # Resume from partial download if .tmp exists
                        existing_size = tmp.stat().st_size if tmp.exists() else 0
                        req_headers = {}
                        if existing_size > 0:
                            req_headers["Range"] = f"bytes={existing_size}-"
                            logging.info(f"Resuming download from byte {existing_size}: {candidate_url}")
                        else:
                            logging.info(f"Downloading: {candidate_url}")

                        with session.get(
                            candidate_url,
                            stream=True,
                            allow_redirects=True,
                            timeout=(10, 300),
                            headers=req_headers,
                        ) as response:
                            response.raise_for_status()

                            if response.status_code == 206:
                                mode = "ab"
                                remaining = int(response.headers.get("content-length", 0) or 0)
                                total_size = existing_size + remaining
                            else:
                                mode = "wb"
                                existing_size = 0
                                total_size = int(response.headers.get("content-length", 0) or 0)

                            logging.info(f"Total size: {total_size} bytes")

                            downloaded = existing_size
                            with (
                                open(tmp, mode) as f,
                                tqdm(
                                    desc=dest.name,
                                    total=total_size or None,
                                    initial=existing_size,
                                    unit="B",
                                    unit_scale=True,
                                    unit_divisor=1024,
                                    disable=not progress_bar,
                                ) as bar,
                            ):
                                for chunk in response.iter_content(chunk_size=8192):
                                    if not chunk:
                                        continue
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    bar.update(len(chunk))

                                    if _progress_dict is not None and _task_id is not None:
                                        _progress_dict[_task_id] = {
                                            "progress": downloaded,
                                            "total": total_size,
                                        }
                    break
                except Exception as candidate_error:
                    errors.append((candidate_url, candidate_error))
                    continue
            else:
                attempted = ", ".join(url for url, _ in errors)
                raise RuntimeError(f"Failed to download from all candidate URLs: {attempted}") from errors[-1][1]

            # Validate size on disk
            if total_size and tmp.stat().st_size != total_size:
                raise RuntimeError(f"Download incomplete for {url}: got {tmp.stat().st_size} of {total_size} bytes")

            # Validate checksum if provided
            if checksum:
                algo, expected = checksum.split(":", 1)
                h_check = hashlib.new(algo)
                with open(tmp, "rb") as f:
                    while True:
                        block = f.read(65536)
                        if not block:
                            break
                        h_check.update(block)
                if h_check.hexdigest() != expected:
                    tmp.unlink()
                    raise ValueError(f"Checksum mismatch for {url}: expected {expected}, got {h_check.hexdigest()}")

            tmp.replace(dest)  # atomic rename
            logging.info(f"Download finished: {dest}")
            return dest

        except Exception as e:
            # Keep .tmp for resumable errors (network, timeout, incomplete).
            # Only delete on checksum/validation failure (ValueError) where
            # the data is known to be corrupt and cannot be resumed.
            if isinstance(e, ValueError):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
            logging.error(f"Error downloading {url}: {e}")
            raise
        finally:
            # Remove lock file so download dir stays clean (FileLock may leave it on some platforms / crash)
            try:
                if lock.exists():
                    lock.unlink()
            except Exception:
                pass


def load_from_tsfile_to_dataframe(
    full_file_path_and_name,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    """Load data from a .ts file into a Pandas DataFrame.
    Credit to https://github.com/sktime/sktime/blob/7d572796ec519c35d30f482f2020c3e0256dd451/sktime/datasets/_data_io.py#L379
    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (
        X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced
       with prior to parsing.
    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
    """
    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_int = None
    prev_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0
    # Parse the file
    with open(full_file_path_and_name, encoding="utf-8") as file:
        for line in file:
            # Strip white space from start/end of line and change to
            # lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this
                # function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError("timestamps tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise OSError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise OSError("univariate tag requires an associated Boolean  value")
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise OSError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise OSError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError("classlabel tag requires an associated Boolean  value")
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise OSError("if the classlabel tag is true then class values must be supplied")
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    if data_started:
                        raise OSError("metadata must come before data")
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise OSError("targetlabel tag requires an associated Boolean value")
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise OSError("invalid targetlabel value")
                    if token_len > 2:
                        raise OSError(
                            "targetlabel tag should not be accompanied with info "
                            "apart from true/false, but found "
                            f"{tokens}"
                        )
                    has_class_labels_tag = True
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise OSError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise OSError("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been
                # parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    if (
                        not has_problem_name_tag
                        or not has_timestamps_tag
                        or not has_univariate_tag
                        or not has_class_labels_tag
                        or not has_data_tag
                    ):
                        raise OSError("a full set of metadata has not been provided before the data")
                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)
                    # Check if we are dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split
                        # line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False
                        timestamp_for_dim = []
                        values_for_dimension = []
                        this_line_num_dim = 0
                        line_len = len(line)
                        char_num = 0
                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1
                            # See if there is any more data to read in or if
                            # we should validate that read thus far
                            if char_num < line_len:
                                # See if we have an empty dimension (i.e. no
                                # values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dim + 1):
                                        instance_list.append([])
                                    instance_list[this_line_num_dim].append(pd.Series(dtype="object"))
                                    this_line_num_dim += 1
                                    has_another_value = False
                                    has_another_dimension = True
                                    timestamp_for_dim = []
                                    values_for_dimension = []
                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and class_labels:
                                        class_val = line[char_num:].strip()
                                        if class_val not in class_label_list:
                                            raise OSError(
                                                "the class value '"
                                                + class_val
                                                + "' on line "
                                                + str(line_num + 1)
                                                + " is not "
                                                "valid"
                                            )
                                        class_val_list.append(class_val)
                                        char_num = line_len
                                        has_another_value = False
                                        has_another_dimension = False
                                        timestamp_for_dim = []
                                        values_for_dimension = []
                                    else:
                                        # Read in the data contained within
                                        # the next tuple
                                        if line[char_num] != "(" and not class_labels:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not "
                                                "start "
                                                "with a "
                                                "'('"
                                            )
                                        char_num += 1
                                        tuple_data = ""
                                        while char_num < line_len and line[char_num] != ")":
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if char_num >= line_len or line[char_num] != ")":
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " does "
                                                "not end"
                                                " with a "
                                                "')'"
                                            )
                                        # Read in any spaces immediately
                                        # after the current tuple
                                        char_num += 1
                                        while char_num < line_len and str.isspace(line[char_num]):
                                            char_num += 1

                                        # Check if there is another value or
                                        # dimension to process after this tuple
                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False
                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False
                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True
                                        char_num += 1
                                        # Get the numeric value for the
                                        # tuple by reading from the end of
                                        # the tuple data backwards to the
                                        # last comma
                                        last_comma_index = tuple_data.rfind(",")
                                        if last_comma_index == -1:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that has "
                                                "no comma inside of it"
                                            )
                                        try:
                                            value = tuple_data[last_comma_index + 1 :]
                                            value = float(value)
                                        except ValueError:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that does "
                                                "not have a valid numeric "
                                                "value"
                                            )
                                        # Check the type of timestamp that
                                        # we have
                                        timestamp = tuple_data[0:last_comma_index]
                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False
                                        if not timestamp_is_int:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False
                                        # Make sure that the timestamps in
                                        # the file (not just this dimension
                                        # or case) are consistent
                                        if not timestamp_is_timestamp and not timestamp_is_int:
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '" + timestamp + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        if (
                                            prev_timestamp_was_timestamp is not None
                                            and prev_timestamp_was_timestamp
                                            and not timestamp_is_timestamp
                                        ):
                                            raise OSError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains tuples where the "
                                                "timestamp format is "
                                                "inconsistent"
                                            )
                                        # Store the values
                                        timestamp_for_dim += [timestamp]
                                        values_for_dimension += [value]
                                        #  If this was our first tuple then
                                        #  we store the type of timestamp we
                                        #  had
                                        if prev_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if previous_timestamp_was_int is None and timestamp_is_int:
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (this_line_num_dim + 1):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(timestamp_for_dim)

                                            instance_list[this_line_num_dim].append(
                                                pd.Series(
                                                    index=timestamp_for_dim,
                                                    data=values_for_dimension,
                                                )
                                            )
                                            this_line_num_dim += 1
                                            timestamp_for_dim = []
                                            values_for_dimension = []
                            elif has_another_value:
                                raise OSError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line " + str(line_num + 1) + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise OSError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line " + str(line_num + 1) + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(pd.Series(dtype=np.float32))
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise OSError(
                                        "line " + str(line_num + 1) + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise OSError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise OSError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ':' while it "
                                "should list a class value"
                            )
                        elif has_another_dimension and not class_labels:
                            if len(instance_list) < (this_line_num_dim + 1):
                                instance_list.append([])
                            instance_list[this_line_num_dim].append(pd.Series(dtype="object"))
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if not has_another_value and num_dimensions != this_line_num_dim:
                            raise OSError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise OSError("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (
                        # that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)
                            if class_labels:
                                num_dimensions -= 1
                            for _dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False
                        # See how many dimensions that the case whose data
                        # in represented in this line has
                        this_line_num_dim = len(dimensions)
                        if class_labels:
                            this_line_num_dim -= 1
                        # All dimensions should be included for all series,
                        # even if they are empty
                        if this_line_num_dim != num_dimensions:
                            raise OSError(
                                "inconsistent number of dimensions. "
                                "Expecting " + str(num_dimensions) + " but have read " + str(this_line_num_dim)
                            )
                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series(dtype="object"))
                        if class_labels:
                            class_val_list.append(dimensions[num_dimensions].strip())
            line_num += 1
    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        if metadata_started and not (
            has_problem_name_tag
            and has_timestamps_tag
            and has_univariate_tag
            and has_class_labels_tag
            and has_data_tag
        ):
            raise OSError("metadata incomplete")

        elif metadata_started and not data_started:
            raise OSError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise OSError("file contained metadata but no data")
        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)
        for dim in range(0, num_dimensions):
            data["dim_" + str(dim)] = instance_list[dim]
        # Check if we should return any associated class labels separately
        if class_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data["class_vals"] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise OSError("empty file")
