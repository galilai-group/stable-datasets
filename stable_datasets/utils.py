import numpy as np
from multiprocessing import Pool, Queue, Lock, Process
from scipy import ndimage
import os
import urllib
import requests
import functools
import shutil

import hashlib
from tqdm import tqdm

import tarfile
import zipfile
import pathlib
from pathlib import Path
import pandas as pd

import torch as ch
from typing import Dict, Union
from PIL import Image
import time
from torch.utils.data import Dataset as TorchDataset


class Dataset(dict):
    # def __getitem__(self, i):
    #     return Image.open(list.__getitem__(self, i)).convert("RGB")

    def __init__(self, path=None, **kwargs):
        if path is None:
            if "AI_DATASETS_ROOT" not in os.environ:
                raise ValueError("path can not be None unless AI_DATASETS_ROOT is set")
            self._path = Path(os.environ["AI_DATASETS_ROOT"])
        else:
            self._path = Path(path)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def md5(self):
        return {}

    @property
    def path(self):
        return self._path

    @property
    def extract(self):
        return []

    @property
    def modalities(self):
        return {}

    @property
    def num_classes(self):
        raise NotImplementedError("You need to define your own num_classes method")

    @property
    def name(self):
        return type(self).__name__

    @property
    def urls(self):
        raise NotImplementedError("You need to define your own urls method")

    @property
    def num_samples(self):
        raise NotImplementedError("You need to define your own num_samples method")

    def __getitem__(self, key):
        if key not in self:
            raise ValueError(
                f"{key} not present.... did you download/load the dataset first?"
            )
        return super().__getitem__(key)

    def download(self):
        """dataset downlading utility"""
        folder = self.path / self.name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, url in self.urls.items():
            download_url(url, folder / filename, self.md5.get(filename, None))
            if filename in self.extract:
                to = os.path.splitext(folder / ("extracted_" + filename))[0]
                extract_file(folder / filename, to)
        return self

    @property
    def load(self):
        raise NotImplementedError("You need to define your own load method")

    def enforce_RGB(self):
        for name, modality in self.modalities.items():
            if modality == "image":
                if isinstance(self[name], np.ndarray):
                    if self[name].shape[-1] == 1:
                        print(f"enforcing Grayscale -> RGB for {name}")
                        self[name] = np.repeat(self[name], 3, axis=-1)
                    elif self[name].ndim == 3:
                        print(f"enforcing Grayscale -> RGB for {name}")
                        self[name] = np.repeat(self[name][..., None], 3, axis=-1)

                else:
                    print(f"enforcing RGB for {name}")
                    self[name] = [x.convert("RGB") for x in self[name]]


def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).
    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type or tuple of type, optional
        required type or types for all elements
    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.
    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        if t == int_types:
            expected_type = "int"  # easier to understand
        elif isinstance(t, tuple):
            expected_type = " or ".join(tt.__name__ for tt in t)
        else:
            expected_type = t.__name__
        raise TypeError(
            "expected a single value or an iterable "
            "of {0}, got {1} instead".format(expected_type, x)
        )

    if len(X) != N:
        raise ValueError(
            "expected a single value or an iterable "
            "with length {0}, got {1} instead".format(N, x)
        )

    return X


def to_numeric_classes(values):
    return np.argmax(values[:, None] == np.unique(values), 1)


def create_cmap(values, colors):
    from matplotlib.pyplot import Normalize
    import matplotlib

    norm = Normalize(min(values), max(values))
    tuples = list(zip(map(norm, values), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    return cmap, norm


def m4a_to_wav(filename):
    m4a_file = "20211210_151013.m4a"
    wav_filename = r"F:\20211210_151013.wav"
    from pydub import AudioSegment

    track = AudioSegment.from_file(m4a_file, format="m4a")
    file_handle = track.export(wav_filename, format="wav")


def patchify_1d(x, window_length, stride):
    """extract patches from a numpy array

    Parameters
    ----------

    x: array-like
        the input data to extract patches from, any shape, the last dimension
        is the one being patched

    window_length: int
        the length of the patches

    stride: int
        the amount of stride (bins separating two consecutive patches

    Returns
    -------

    x_patches: array-like
        the number of patches is put in the pre-last dimension (-2)
    """

    n_windows = (x.shape[-1] - window_length) // stride + 1
    new_x = np.empty(x.shape[:-1] + (n_windows, window_length))
    for n in range(n_windows):
        new_x[..., n, :] = x[..., n * stride : n * stride + window_length]
    return new_x


def patchify_2d(x, window_length, stride):
    # TODO
    return None


def train_test_split(*args, train_size=0.8, stratify=None, seed=None):
    """split given data into two non overlapping sets

    Parameters
    ----------

    *args: inputs
        the sets to be split by the function

    train_size: scalar
        the amount of data to put in the first set, either an integer value
        being the actual number of data to keep, or a ratio (0 to 1 number)

    stratify: array (optional)
        the optimal stratify guide to spit the array s.t. the same proportion
        based on the stratify array is kep in both set based on the proportion
        of the split

    seed: integer (optional)
        the seed for the random number generator for reproducibility

    Returns
    -------

    train_set: list
        returns the train data, the list has the members of *args split

    test_set: list
        returns the test data, the list has the members of *args split

    Example
    -------

    .. code-block:: python

       x = numpy.random.randn(100, 4)
       y = numpy.random.randn(100)

       train, test = train_test_split(x, y, train_size=0.5)
       print(train[0].shape, train[1].shape)
       # (50, 4) (50,)
       print(test[0].shape, test[1].shape)
       # (50, 4) (50,)


    """
    if stratify is not None:
        train_indices = list()
        test_indices = list()
        for c in set(list(stratify)):
            c_indices = np.where(stratify == c)[0]
            np.random.RandomState(seed=seed).shuffle(c_indices)
            if train_size > 1:
                cutoff = train_size
            else:
                cutoff = int(len(c_indices) * train_size)
            train_indices.append(c_indices[:cutoff])
            test_indices.append(c_indices[cutoff:])
        train_indices = np.concatenate(train_indices, 0)
        test_indices = np.concatenate(test_indices, 0)
    else:
        indices = np.random.RandomState(seed=seed).permutation(len(args[0]))
        if train_size > 1:
            assert type(train_size) == int
            cutoff = train_size
        else:
            cutoff = int(len(args[0]) * train_size)
        print(cutoff)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
    train_set = [arg[train_indices] for arg in args]
    test_set = [arg[test_indices] for arg in args]
    if len(args) == 1:
        return train_set[0], test_set[0]
    return train_set, test_set


class batchify:
    def __init__(
        self,
        *args,
        batch_size,
        option="random",
        load_func=None,
        extra_process=0,
        n_batches=None,
    ):
        """generator to iterate though mini-batches

        Parameters
        ----------

        load_func: None or list of func
            same length as the number of args. A function is called on a single
            datum and it can be used to apply some normalization but its main
            goal is to load files if the args were list of filenames

        extra_processes: int (optional)
            if there is no load_func then extra process is useless

        n_batches: int (optional)
            the number of batches to produce, only used if option is random, if
            not given it is taken to be the length of the data divided by the
            batch_size

        Returns
        -------

        *batch_args: list
            the iterator containing the batch values
            of each arg in args

        Example
        -------

        .. code-block:: python

        for x, y in batchify(X, Y):
            train(x, y)

        """

        self.n_batches = n_batches or len(args[0]) // batch_size
        self.args = args
        self.start_index = 0
        self.option = option
        self.batch_size = batch_size
        self.extra_process = extra_process
        self.terminate = False

        if option == "random_see_all":
            self.permutation = np.random.permutation(len(args[0]))

        # set up load function
        if load_func is None:
            self.load_func = (None,) * len(args)
        else:
            self.load_func = []
            for i in range(len(load_func)):
                if load_func[i] is None:
                    self.load_func.append(None)
                elif extra_process == 0:

                    def fn(args, queue, f=load_func[i]):
                        result = np.asarray([f(arg) for arg in args])
                        queue.put(result)

                    self.load_func.append(fn)
                else:

                    def fn(lock, data, queue, f=load_func[i]):
                        lock.acquire()
                        with Pool(processes=extra_process) as pool:
                            result = pool.map(f, data)
                        queue.put(np.asarray(result))
                        lock.release()

                    self.load_func.append(fn)
        assert np.prod([len(args[0]) == len(arg) for arg in args[1:]])

        self.queues = [Queue() for f in self.load_func]
        self.locks = [Lock() for f in self.load_func]

        # launch the first batch straight away
        batch = self.get_batch()
        self.launch_process(batch)

    def chunk(self, items, n):
        for i in range(0, len(items), n):
            yield items[i : i + n]

    def launch_process(self, batch):
        for b, lock, load_func, queue in zip(
            batch, self.locks, self.load_func, self.queues
        ):
            if load_func is None:
                queue.put(b)
            elif load_func is not None and self.extra_process == 0:
                load_func(b, queue)
            elif load_func is not None and self.extra_process > 0:
                # first chunk for each process and launch a process
                p = Process(target=load_func, args=(lock, b, queue))
                p.start()

    def __iter__(self):
        return self

    def get_batch(self):
        indices = (self.start_index, self.start_index + self.batch_size)

        # check if we exhausted the samples
        if self.option == "random":
            if indices[1] > self.batch_size * self.n_batches:
                raise StopIteration()
        elif indices[1] > len(self.args[0]):
            raise StopIteration()

        # proceed to get the data
        if self.option == "random_see_all":
            perm = self.permutation[indices[0] : indices[1]]
            batch = [
                arg[perm] if hasattr(arg, "shape") else [arg[i] for i in perm]
                for arg in self.args
            ]
        elif self.option == "continuous":
            batch = [arg[indices[0] : indices[1]] for arg in self.args]
        elif self.option == "random":
            perm = np.random.randint(0, len(self.args[0]), self.batch_size)
            batch = [
                arg[perm] if hasattr(arg, "shape") else [arg[i] for i in perm]
                for arg in self.args
            ]
        return batch

    def __next__(self):
        if self.terminate:
            raise StopIteration()

        # we prepare the next batch if possible
        try:
            self.start_index += self.batch_size
            batch = self.get_batch()
            self.launch_process(batch)
        except StopIteration:
            self.terminate = True

        if len(self.queues) == 1:
            return self.queues[0].get()
        else:
            return tuple([queue.get() for queue in self.queues])

    def apply_func(self, batch):
        # now check if needs to apply a load func
        for i, load_func in enumerate(self.load_func):
            if load_func is not None:
                batch[i] = load_func(batch[i])


def resample_images(
    images,
    target_shape,
    ratio="same",
    order=1,
    mode="nearest",
    data_format="channels_first",
):
    if data_format == "channels_first":
        output_images = np.zeros(
            (len(images), images[0].shape[0]) + target_shape,
            dtype=images[0].dtype,
        )
    else:
        output_images = np.zeros(
            (len(images),) + target_shape + (images[0].shape[-1],),
            dtype=images[0].dtype,
        )

    for i, image in enumerate(images):
        # the first step is to resample to fit it into the target shape
        if data_format != "channels_first":
            image = image.transpose(2, 0, 1)

        if ratio == "same":
            width_change = target_shape[1] / image.shape[2]
            height_change = target_shape[0] / image.shape[1]
            change = min(width_change, height_change)
        x = np.linspace(0, image.shape[1] - 1, int(image.shape[1] * change))
        y = np.linspace(0, image.shape[2] - 1, int(image.shape[2] * change))
        coordinates = np.stack(np.meshgrid(x, y))
        coordinates = np.stack([coordinates[0].reshape(-1), coordinates[1].reshape(-1)])

        new_image = np.stack(
            [
                ndimage.map_coordinates(channel, coordinates, order=order, mode=mode)
                .reshape((len(y), len(x)))
                .T
                for channel in image
            ]
        )
        # now we position this image into the output
        middle_height = (target_shape[0] - len(x)) // 2
        middle_width = (target_shape[1] - len(y)) // 2
        if data_format != "channels_first":
            output_images[
                i,
                middle_height : middle_height + len(x),
                middle_width : middle_width + len(y),
                :,
            ] = new_image.transpose(1, 2, 0)
        else:
            output_images[
                i,
                :,
                middle_height : middle_height + len(x),
                middle_width : middle_width + len(y),
            ] = new_image
    return output_images


def download_url(url: str, filename: str, md5_checksum: str = None):
    if "drive.google.com" in url:
        import gdown

        gdown.download(url, str(filename), quiet=False)
        return

    target = Path(filename).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if target.is_file():
        if md5_checksum is not None and checksum(target, md5_checksum):
            print(f"{filename} already downloaded with valid checksum")
            return
        else:
            start = target.stat().st_size
    r = requests.get(
        url,
        headers={"Range": f"bytes={start}-"},
        stream=True,
        verify=False,
        allow_redirects=True,
    )
    file_size = int(r.headers.get("Content-Length", 0))

    if file_size and file_size == start:
        print(
            f"File {filename} was already downloaded from URL {url} and has right size"
        )
    elif file_size and start > file_size:
        print("File is bigger than expected...")
    else:
        try:
            os.open(str(target) + ".lock", os.O_CREAT | os.O_EXCL)
            desc = str(target)
            if file_size == 0:
                desc += " (Unknown total file size)"
            else:
                desc += f" ({file_size}b)"
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
                with target.open("wb" if start == 0 else "ab") as f:
                    shutil.copyfileobj(r_raw, f)
            os.remove(str(target) + ".lock")
        except Exception as e:
            print(e)
            while os.path.isfile(str(target) + ".lock"):
                print(f"waiting for {target.name} to be downloaded by another process")
                time.sleep(1)

    if md5_checksum is None:
        current = hashlib.md5(open(target, "rb").read()).hexdigest()
        print(f"A md5 checksum was not given, please use {current} for {target}")
    else:
        assert checksum(target, md5_checksum)
        print("Downloaded file matches given md5 checksum")


def checksum(target, md5):
    observed = hashlib.md5(open(target, "rb").read()).hexdigest()
    if observed == md5:
        return True
    else:
        print(f"Expected {md5} but is {observed}")


def track_progress(members, total):
    for member in tqdm(members, total=total, desc="Extracting..."):
        yield member


def extract_file(filename, target):
    if os.path.isfile(str(target) + ".done"):
        print("already extracted")
        return
    ext = pathlib.Path(filename).suffix
    try:
        os.open(str(target) + ".lock", os.O_CREAT | os.O_EXCL)
        if Path(target).is_dir():
            print("Already extracted (but not verified) leaving")
            return
        if ext in [".tgz", ".tar"] or str(filename)[-7:] == ".tar.gz":
            tgz = ext == ".tgz" or str(filename)[-7:] == ".tar.gz"
            with tarfile.open(filename, "r:gz" if tgz else "r") as tarball:
                tarball.extractall(path=target, members=track_progress(tarball, None))
        elif ext == ".zip":
            with zipfile.ZipFile(filename) as zip_file:
                for member in tqdm(zip_file.namelist(), desc="Extracting "):
                    if os.path.exists(target + r"/" + member) or os.path.isfile(
                        target + r"/" + member
                    ):
                        continue
                    zip_file.extract(member, target)
        os.remove(str(target) + ".lock")
        os.open(str(target) + ".done", os.O_CREAT)
    except Exception as e:
        print(e)
        while os.path.isfile(str(target) + ".lock"):
            print(f"{filename} already being extracted, waiting...")
            time.sleep(3)


def tolist_recursive(array):
    if isinstance(array, np.ndarray):
        return tolist_recursive(array.tolist())
    elif isinstance(array, list):
        return [tolist_recursive(item) for item in array]
    elif isinstance(array, tuple):
        return tuple(tolist_recursive(item) for item in array)
    else:
        return array


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
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
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
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError("problemname tag requires an associated value")
                    # problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "timestamps tag requires an associated Boolean " "value"
                        )
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise IOError("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len != 2:
                        raise IOError(
                            "univariate tag requires an associated Boolean  " "value"
                        )
                    elif tokens[1] == "true":
                        # univariate = True
                        pass
                    elif tokens[1] == "false":
                        # univariate = False
                        pass
                    else:
                        raise IOError("invalid univariate value")
                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise IOError("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError(
                            "classlabel tag requires an associated Boolean  " "value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid classLabel value")
                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise IOError(
                            "if the classlabel tag is true then class values "
                            "must be supplied"
                        )
                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    if data_started:
                        raise IOError("metadata must come before data")
                    tokens = line.split(" ")
                    token_len = len(tokens)
                    if token_len == 1:
                        raise IOError(
                            "targetlabel tag requires an associated Boolean value"
                        )
                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise IOError("invalid targetlabel value")
                    if token_len > 2:
                        raise IOError(
                            "targetlabel tag should not be accompanied with info "
                            "apart from true/false, but found "
                            f"{tokens}"
                        )
                    has_class_labels_tag = True
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise IOError("data tag should not have an associated value")
                    if data_started and not metadata_started:
                        raise IOError("metadata must come before data")
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
                        raise IOError(
                            "a full set of metadata has not been provided "
                            "before the data"
                        )
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
                                    instance_list[this_line_num_dim].append(
                                        pd.Series(dtype="object")
                                    )
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
                                            raise IOError(
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
                                            raise IOError(
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
                                        while (
                                            char_num < line_len
                                            and line[char_num] != ")"
                                        ):
                                            tuple_data += line[char_num]
                                            char_num += 1
                                        if (
                                            char_num >= line_len
                                            or line[char_num] != ")"
                                        ):
                                            raise IOError(
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
                                        while char_num < line_len and str.isspace(
                                            line[char_num]
                                        ):
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
                                            raise IOError(
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
                                            raise IOError(
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
                                        if (
                                            not timestamp_is_timestamp
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
                                                "dimension "
                                                + str(this_line_num_dim + 1)
                                                + " on line "
                                                + str(line_num + 1)
                                                + " contains a tuple that "
                                                "has an invalid timestamp '"
                                                + timestamp
                                                + "'"
                                            )
                                        if (
                                            previous_timestamp_was_int is not None
                                            and previous_timestamp_was_int
                                            and not timestamp_is_int
                                        ):
                                            raise IOError(
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
                                            raise IOError(
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
                                        if (
                                            prev_timestamp_was_timestamp is None
                                            and timestamp_is_timestamp
                                        ):
                                            prev_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False

                                        if (
                                            previous_timestamp_was_int is None
                                            and timestamp_is_int
                                        ):
                                            prev_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                        # See if we should add the data for
                                        # this dimension
                                        if not has_another_value:
                                            if len(instance_list) < (
                                                this_line_num_dim + 1
                                            ):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamp_for_dim = pd.DatetimeIndex(
                                                    timestamp_for_dim
                                                )

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
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ',' that "
                                    "is not followed by "
                                    "another tuple"
                                )
                            elif has_another_dimension and class_labels:
                                raise IOError(
                                    "dimension " + str(this_line_num_dim + 1) + " on "
                                    "line "
                                    + str(line_num + 1)
                                    + " ends with a ':' while "
                                    "it should list a class "
                                    "value"
                                )
                            elif has_another_dimension and not class_labels:
                                if len(instance_list) < (this_line_num_dim + 1):
                                    instance_list.append([])
                                instance_list[this_line_num_dim].append(
                                    pd.Series(dtype=np.float32)
                                )
                                this_line_num_dim += 1
                                num_dimensions = this_line_num_dim
                            # If this is the 1st line of data we have seen
                            # then note the dimensions
                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dim
                                if num_dimensions != this_line_num_dim:
                                    raise IOError(
                                        "line "
                                        + str(line_num + 1)
                                        + " does not have the "
                                        "same number of "
                                        "dimensions as the "
                                        "previous line of "
                                        "data"
                                    )
                        # Check that we are not expecting some more data,
                        # and if not, store that processed above
                        if has_another_value:
                            raise IOError(
                                "dimension "
                                + str(this_line_num_dim + 1)
                                + " on line "
                                + str(line_num + 1)
                                + " ends with a ',' that is "
                                "not followed by another "
                                "tuple"
                            )
                        elif has_another_dimension and class_labels:
                            raise IOError(
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
                            instance_list[this_line_num_dim].append(
                                pd.Series(dtype="object")
                            )
                            this_line_num_dim += 1
                            num_dimensions = this_line_num_dim
                        # If this is the 1st line of data we have seen then
                        # note the dimensions
                        if (
                            not has_another_value
                            and num_dimensions != this_line_num_dim
                        ):
                            raise IOError(
                                "line " + str(line_num + 1) + " does not have the same "
                                "number of dimensions as the "
                                "previous line of data"
                            )
                        # Check if we should have class values, and if so
                        # that they are contained in those listed in the
                        # metadata
                        if class_labels and len(class_val_list) == 0:
                            raise IOError("the cases have no associated class values")
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
                            raise IOError(
                                "inconsistent number of dimensions. "
                                "Expecting "
                                + str(num_dimensions)
                                + " but have read "
                                + str(this_line_num_dim)
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
            raise IOError("metadata incomplete")

        elif metadata_started and not data_started:
            raise IOError("file contained metadata but no data")

        elif metadata_started and data_started and len(instance_list) == 0:
            raise IOError("file contained metadata but no data")
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
        raise IOError("empty file")


@ch.jit.script
def base_two(x: ch.Tensor, bits: int):
    with ch.no_grad():
        mask = 2 ** ch.arange(bits).to(x.device, x.dtype)
        return x.view(-1, 1).bitwise_and(mask).ne_(0).byte()


class TensorDataset(TorchDataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        if self.transform:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.X)


def dataset_to_lightning(
    train,
    val=None,
    test=None,
    num_workers=0,
    create_val=0,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    **kwargs,
):
    import lightning.pytorch as pl

    class DataModule(pl.LightningDataModule):
        def __init__(
            self,
            fn,
            path,
            batch_size,
            create_val,
            num_workers,
        ):
            super().__init__()
            self.fn = fn
            self.path = path
            self.batch_size = batch_size
            self.create_val = create_val
            self.num_workers = num_workers
            self.train_transform = train_transform
            self.val_transform = val_transform
            self.test_transform = test_transform

        def setup(self, stage: str):
            if hasattr(self, "train"):
                return
            dataset = self.fn(self.path)
            self.train = MyDataset(
                dataset["train"]["X"],
                dataset["train"]["y"],
                transform=self.train_transform,
            )
            if val:
                self.val = MyDataset(
                    dataset["val"]["X"],
                    dataset["val"]["y"],
                    transform=self.val_transform,
                )
            if "test" in dataset.keys():
                self.test = MyDataset(
                    dataset["test"]["X"],
                    dataset["test"]["y"],
                    transform=self.val_transform,
                )

        def train_dataloader(self):
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )

        def val_dataloader(self):
            return DataLoader(self.val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.test, batch_size=self.batch_size)

        def predict_dataloader(self):
            return DataLoader(self.predict, batch_size=self.batch_size)

    return DataModule(
        fn, path, batch_size, create_val, num_workers, train_transform, val_transform
    )
