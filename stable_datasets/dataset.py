from pathlib import Path, PosixPath
from typing import Union
from .utils import download_dataset
import time


class Dataset:
    def __init__(self, path: Union[str, PosixPath], download: bool = True):
        self._path = Path(path)
        self._download = download

    def prepare(self):
        if self._download:
            download_dataset(self.name, self.urls, path=self._path)

        t0 = time.time()
        self.load()
        print(f"Dataset {self.name} loaded in{0:.2f}s.".format(time.time() - t0))

    def download(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
