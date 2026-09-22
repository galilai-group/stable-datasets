"""Render AID/RESISC45 teasers from five real classes without downloading full ZIPs.

Run from the repository root:
    PYTHONPATH=. python docs/source/datasets/teasers/generate_remote_sensing_teasers.py
"""

import argparse
import io
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import matplotlib
import requests
from PIL import Image

from stable_datasets.images.aid import AID
from stable_datasets.images.resisc45 import RESISC45


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class RemoteZip(io.RawIOBase):
    """Small seekable HTTP Range adapter for Python's ZIP reader."""

    def __init__(self, url):
        super().__init__()
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.url = response.url
        self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.buffer_start = 0
        self.buffer = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if self.position < 0:
            raise ValueError("Cannot seek before the start of the archive.")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        buffer_offset = self.position - self.buffer_start
        if buffer_offset < 0 or buffer_offset + size > len(self.buffer):
            end = min(self.size - 1, self.position + max(size, 128 * 1024) - 1)
            response = self.session.get(
                self.url,
                headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
                timeout=120,
            )
            response.raise_for_status()
            expected_range = f"bytes {self.position}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                raise OSError("Server did not honor the requested HTTP byte range.")
            self.buffer_start = self.position
            self.buffer = response.content
            buffer_offset = 0
        result = self.buffer[buffer_offset : buffer_offset + size]
        self.position += len(result)
        return result

    def close(self):
        self.session.close()
        super().close()


def generate_teaser(dataset_class, expected_count, native_size, output_path):
    labels = dataset_class._labels()
    samples = []
    with RemoteZip(dataset_class.SOURCE["assets"]["train"]) as remote:
        with zipfile.ZipFile(remote) as archive:
            paths = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".jpg") and "__MACOSX" not in PurePosixPath(name).parts
            )
            counts = Counter(PurePosixPath(name).parent.name for name in paths)
            if len(paths) != expected_count or sorted(counts) != labels:
                raise ValueError(f"Unexpected archive contents: {counts}")
            print(f"{dataset_class.__name__}: {len(paths)} images, {len(counts)} classes", flush=True)
            for label in labels[:5]:
                name = next(name for name in paths if PurePosixPath(name).parent.name == label)
                with Image.open(io.BytesIO(archive.read(name))) as original:
                    if original.size != native_size:
                        raise ValueError(f"Unexpected native image dimensions for {name}: {original.size}")
                    samples.append((label, original.convert("RGB")))
                print(f"  {name}: {native_size}", flush=True)

    figure, axes = plt.subplots(1, len(samples), figsize=(7.5, 1.75))
    for axis, (label, image) in zip(axes, samples):
        axis.imshow(image)
        axis.set_title(label.replace("_", " "), fontsize=8, fontweight="bold", pad=5)
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.85, wspace=0.05)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["aid", "resisc45", "all"], default="all")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    if args.dataset in {"aid", "all"}:
        generate_teaser(AID, 10000, (600, 600), directory / "aid_teaser.png")
    if args.dataset in {"resisc45", "all"}:
        generate_teaser(RESISC45, 31500, (256, 256), directory / "resisc45_teaser.png")
