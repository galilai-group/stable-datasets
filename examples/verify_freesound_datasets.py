"""Check the FSD50K and FSDnoisy18k builders against the real Zenodo files without downloading the audio.

The small metadata archives are downloaded and checked against the md5 checksums the builders declare.
The large audio archives (25 GB / 9.5 GB) are opened remotely with HTTP range requests, so only their
central directories and a few clips are transferred (~150 MB in total). For every split the builder's own
``_generate_examples`` then

* checks that every clip listed in the official ground truth is present in the archive, and
* decodes the first ``--rows`` clips, which are checked for sample rate, length, labels and licenses.

Usage::

    python examples/verify_freesound_datasets.py [--rows 25] [--out verification.json]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from stable_datasets.timeseries import FSD50K, FSDnoisy18k
from stable_datasets.timeseries.fsd50k import _DEV_AUDIO, _EVAL_AUDIO


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only file over HTTP range requests with a small read-ahead buffer."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.requests, self.bytes_fetched = url, block, 0, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
        self._buf_start, self._buf = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self._buf_start <= self.pos and self.pos + n <= self._buf_start + len(self._buf)):
            end = min(self.size, self.pos + max(n, self.block)) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            with urllib.request.urlopen(request) as response:
                self._buf = response.read()
            self._buf_start = self.pos
            self.requests += 1
            self.bytes_fetched += len(self._buf)
        offset = self.pos - self._buf_start
        self.pos += n
        return self._buf[offset : offset + n]


def _download_checked(info, directory: Path) -> Path:
    algo, expected = info.checksum.split(":", 1)
    path = directory / info.filename
    with urllib.request.urlopen(info.url) as response:
        path.write_bytes(response.read())
    actual = hashlib.new(algo, path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{info.filename}: {algo} {actual} != declared {expected}")
    return path


def _builder(cls, config_name="native"):
    builder = object.__new__(cls)
    cls.__init__(builder, config_name=config_name)
    return builder


def _check_rows(generator, n_rows: int, label_key: str, n_classes: int) -> dict:
    rows, t0 = [], time.perf_counter()
    for _, example in generator:  # the generator first checks the whole listing against the archive
        series = np.asarray(example["series"])
        labels = example[label_key] if isinstance(example[label_key], list) else [example[label_key]]
        assert series.ndim == 2 and series.shape[1] == 1 and series.dtype == np.float32 and len(series) > 0
        assert example["sample_rate"] == 44100
        assert np.abs(series).max() <= 32768
        assert labels and all(0 <= label < n_classes for label in labels)
        assert example["license"].startswith("http://creativecommons.org/")
        rows.append({"filename": example["filename"], "seconds": round(len(series) / 44100, 2), "labels": labels})
        if len(rows) == n_rows:
            break
    return {"decoded_rows": len(rows), "seconds": time.perf_counter() - t0, "examples": rows[:3]}


def verify_fsd50k(tmp: Path, n_rows: int) -> dict:
    assets = FSD50K.SOURCE["assets"]
    gt = _download_checked(assets["ground_truth"], tmp)
    meta = _download_checked(assets["metadata"], tmp)
    dev = [HTTPRangeFile(assets[name].url) for name, _ in _DEV_AUDIO]
    ev = [HTTPRangeFile(assets[name].url) for name, _ in _EVAL_AUDIO]
    builder = _builder(FSD50K)
    out = {}
    for split, parts in (("train", dev), ("val", dev), ("eval", ev)):
        gen = builder._generate_examples(audio_parts=parts, ground_truth_path=gt, metadata_path=meta, split=split)
        out[split] = _check_rows(gen, n_rows, "labels", 200)
    out["bytes_fetched_mb"] = round(sum(f.bytes_fetched for f in dev + ev) / 1e6, 1)
    return out


def verify_fsdnoisy18k(tmp: Path, n_rows: int) -> dict:
    assets = FSDnoisy18k.SOURCE["assets"]
    meta = _download_checked(assets["meta"], tmp)
    doc = _download_checked(assets["doc"], tmp)
    builder = _builder(FSDnoisy18k)
    out = {}
    for split, asset in (("train", "train_audio"), ("test", "test_audio")):
        remote = HTTPRangeFile(assets[asset].url)
        gen = builder._generate_examples(audio_path=remote, meta_path=meta, doc_path=doc, split=split)
        out[split] = _check_rows(gen, n_rows, "label", 20)
        out[f"{split}_bytes_fetched_mb"] = round(remote.bytes_fetched / 1e6, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=25, help="clips to decode per split")
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        report = {
            "FSD50K": verify_fsd50k(Path(tmp), args.rows),
            "FSDnoisy18k": verify_fsdnoisy18k(Path(tmp), args.rows),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
