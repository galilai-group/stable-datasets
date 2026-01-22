import multiprocessing
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

import datasets
import numpy as np
import pandas as pd
import rich.progress
from datasets import DownloadConfig
from filelock import FileLock
from loguru import logger as logging
from requests_cache import CachedSession
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm


DEFAULT_CACHE_DIR = "~/.stable_datasets/"


def _default_dest_folder() -> Path:
    """Default folder where files are saved."""
    return Path(os.path.expanduser(DEFAULT_CACHE_DIR)) / "downloads"


def _default_processed_cache_dir() -> Path:
    """Default folder where processed datasets (Arrow files) are cached."""
    return Path(os.path.expanduser(DEFAULT_CACHE_DIR)) / "processed"


class BaseDatasetBuilder(datasets.GeneratorBasedBuilder):
    """
    Base class for stable-datasets that enables direct dataset loading.
    """

    # Subclasses must define:
    # - VERSION: datasets.Version
    #
    # For dataset provenance / downloads, subclasses can either:
    # - define a class attribute SOURCE (static), or
    # - override _source(self) to compute it at runtime (e.g. from self.config)
    VERSION: datasets.Version
    SOURCE: Mapping

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

        # VERSION must exist and be a datasets.Version
        if not hasattr(cls, "VERSION"):
            raise TypeError(f"{cls.__name__} must define a class attribute VERSION = datasets.Version('x.y.z').")
        if not isinstance(getattr(cls, "VERSION"), datasets.Version):
            raise TypeError(f"{cls.__name__}.VERSION must be a datasets.Version instance.")

        # Enforce that a source is provided either statically (SOURCE) or dynamically (_source override).
        has_static_source = hasattr(cls, "SOURCE")
        has_dynamic_source = cls._source is not BaseDatasetBuilder._source  # overridden
        if not (has_static_source or has_dynamic_source):
            raise TypeError(
                f"{cls.__name__} must define SOURCE = {{...}} or override _source(self) to compute it at runtime."
            )

        # Freeze static SOURCE at class creation time.
        if has_static_source:
            cls.SOURCE = cls._freeze(getattr(cls, "SOURCE"))

        # If subclass overrides _source(), wrap it so the returned mapping is frozen/immutable.
        if has_dynamic_source and not getattr(cls._source, "_stable_datasets_freezes_source", False):
            original = cls._source

            def _wrapped_source(self):
                source = original(self)
                return BaseDatasetBuilder._freeze(source)

            _wrapped_source._stable_datasets_freezes_source = True  # type: ignore[attr-defined]
            cls._source = _wrapped_source  # type: ignore[method-assign]

    def _source(self) -> Mapping:
        """
        Return dataset provenance / download configuration.

        Default: uses a class attribute SOURCE (frozen into an immutable Mapping).
        Override in subclasses when the source depends on runtime config (e.g. self.config.variant).
        """
        if not hasattr(self.__class__, "SOURCE"):
            raise TypeError(f"{self.__class__.__name__} does not define SOURCE and did not override _source().")
        return getattr(self.__class__, "SOURCE")

    @staticmethod
    def _validate_source(source: Mapping) -> None:
        if not isinstance(source, Mapping):
            raise TypeError("source must be a mapping.")

        # Required for provenance
        if "homepage" not in source or source["homepage"] is None or not isinstance(source["homepage"], str):
            raise TypeError("SOURCE['homepage'] must be a string and must be present.")
        if "citation" not in source or source["citation"] is None or not isinstance(source["citation"], str):
            raise TypeError("SOURCE['citation'] must be a string and must be present.")

        # Required for downloads (even if a dataset overrides _split_generators).
        if "assets" not in source or not isinstance(source["assets"], Mapping):
            raise TypeError("SOURCE must contain a mapping-valued 'assets' key.")

    def _split_generators(self, dl_manager):
        """
        Default split generator implementation.

        Most stable-datasets follow the pattern "one downloadable file per split", expressed
        via `SOURCE["assets"]`. Datasets with different layouts can override this method.
        """
        source = self._source()
        if not isinstance(source, Mapping):
            raise TypeError(f"{self.__class__.__name__}._source() must return a mapping.")
        self._validate_source(source)

        assets = source["assets"]
        if len(assets) == 0:
            raise ValueError(f"{self.__class__.__name__}.SOURCE['assets'] is empty; cannot infer splits.")

        split_names = list(assets.keys())
        ordered_urls = [assets[s] for s in split_names]

        # stable-datasets standardizes on our local bulk downloader (not HF dl_manager).
        # Deduplicate URLs to avoid redundant downloads for datasets where all splits share a single file.
        unique_urls = list(dict.fromkeys(ordered_urls))
        download_dir = getattr(self, "_raw_download_dir", None)
        if download_dir is None:
            download_dir = _default_dest_folder()
        unique_paths = bulk_download(unique_urls, dest_folder=download_dir)
        url_to_path = dict(zip(unique_urls, unique_paths))
        local_paths = [url_to_path[u] for u in ordered_urls]

        split_to_path = dict(zip(split_names, local_paths))

        name_map = {
            "train": datasets.Split.TRAIN,
            "test": datasets.Split.TEST,
            "val": datasets.Split.VALIDATION,
        }

        return [
            datasets.SplitGenerator(
                name=name_map.get(split_name, split_name),
                gen_kwargs={"data_path": split_to_path[split_name], "split": split_name},
            )
            for split_name in split_names
        ]

    def __new__(cls, *args, split=None, processed_cache_dir=None, download_dir=None, **kwargs):
        """
        Automatically download, prepare, and return the dataset for the specified split.

        Args:
            split: Dataset split to load (e.g., "train", "test", "validation"). If None,
                loads all available splits and returns a DatasetDict.
            processed_cache_dir: Cache directory for processed datasets (Arrow cache). If None,
                defaults to ~/.stable_datasets/processed/.
            download_dir: Directory for raw downloads (ZIP/NPZ/etc). If None, defaults to
                ~/.stable_datasets/downloads/.
            **kwargs: Additional arguments passed to the dataset builder.

        Returns:
            Union[datasets.Dataset, datasets.DatasetDict]: The loaded dataset (single split)
                or a DatasetDict (all splits).
        """
        instance = super().__new__(cls)

        # 1) Decide cache locations
        # Processed cache (Arrow)
        if processed_cache_dir is None:
            processed_cache_dir = str(_default_processed_cache_dir())
        instance._processed_cache_dir = Path(processed_cache_dir)

        # Raw downloads
        if download_dir is None:
            download_dir = str(_default_dest_folder())
        instance._raw_download_dir = Path(download_dir)

        # 2) Initialize builder with our processed cache_dir explicitly
        instance.__init__(*args, cache_dir=str(processed_cache_dir), **kwargs)

        # 2b) Validate dataset SOURCE contract early.
        source = instance._source()
        if not isinstance(source, Mapping):
            raise TypeError(f"{cls.__name__}._source() must return a mapping.")
        cls._validate_source(source)

        # 3) Explicitly tell HF to use our processed cache_dir for any dl_manager downloads
        download_config = DownloadConfig(cache_dir=str(processed_cache_dir))

        instance.download_and_prepare(
            download_config=download_config,
        )

        # 4) Load the split from the same cache_dir
        if split is None:
            result = instance.as_dataset()
        else:
            result = instance.as_dataset(split=split)

        # Expose cache locations on the returned dataset object for convenience.
        # Note: DatasetDict may not allow attribute assignment; ignore if not supported.
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
    urls: Iterable[str],
    dest_folder: str | Path,
    backend: str = "filesystem",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> list[Path]:
    """
    Download multiple files concurrently and return their local paths.

    Args:
        urls: Iterable of URL strings to download.
        dest_folder: Destination folder for downloads.
        backend: requests_cache backend (e.g. "filesystem").
        cache_dir: Cache directory for requests_cache.

    Returns:
        list[Path]: Local file paths in the same order as the input URLs.
    """
    urls = list(urls)
    num_workers = len(urls)
    if num_workers == 0:
        return []

    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    filenames = [os.path.basename(urlparse(url).path) for url in urls]
    results: list[Path] = []

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
                for i in range(num_workers):
                    task_id = filenames[i]
                    future = executor.submit(
                        download,
                        urls[i],
                        dest_folder,
                        backend,
                        cache_dir,
                        False,  # disable per-file tqdm; Rich handles progress
                        _progress,
                        task_id,
                    )
                    futures.append(future)

                rich_tasks = {}

                # update Rich progress while downloads are running
                while not all(f.done() for f in futures):
                    for task_id in list(_progress.keys()):
                        prog = _progress[task_id]
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
            for future in futures:
                results.append(future.result())

    return results


def download(
    url: str,
    dest_folder: str | Path | None = None,
    backend: str = "filesystem",
    cache_dir: str = DEFAULT_CACHE_DIR,
    progress_bar: bool = True,
    _progress_dict=None,
    _task_id=None,
) -> Path:
    """
    Download a single file from a URL with caching and optional progress tracking.

    Args:
        url: URL to download from.
        dest_folder: Destination folder for the downloaded file. If None,
            defaults to ~/.stable_datasets/downloads/.
        backend: requests_cache backend (e.g. "filesystem").
        cache_dir: Cache directory for requests_cache.
        progress_bar: Whether to show a tqdm progress bar (for standalone use).
        _progress_dict: Internal shared dict for bulk_download progress reporting.
        _task_id: Internal task ID key for bulk_download progress reporting.

    Returns:
        Path: Local path to the downloaded file.

    Raises:
        Exception: Any exception from network/file operations is logged and re-raised.
    """
    try:
        if dest_folder is None:
            dest_folder = _default_dest_folder()
        dest_folder = Path(dest_folder)
        dest_folder.mkdir(parents=True, exist_ok=True)

        filename = os.path.basename(urlparse(url).path)
        local_filename = dest_folder / filename
        lock_filename = dest_folder / f"{filename}.lock"

        # prevent concurrent downloads of the same file
        with FileLock(lock_filename):
            session = CachedSession(cache_dir, backend=backend)
            session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
            logging.info(f"Downloading: {url}")

            response = session.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0) or 0)
            logging.info(f"Total size: {total_size} bytes")

            downloaded = 0

            with (
                open(local_filename, "wb") as f,
                tqdm(
                    desc=local_filename.name,
                    total=total_size or None,  # None if size unknown
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

            if total_size and downloaded != total_size:
                logging.error(f"Download incomplete: got {downloaded} of {total_size} bytes for {url}")
            else:
                logging.info(f"Download finished: {local_filename}")

            return local_filename

    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        raise e


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
