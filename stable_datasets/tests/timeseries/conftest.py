"""Shared fixtures for the time-series / audio builder tests."""

import io
import struct
import urllib.request
import zipfile
import zlib

import numpy as np
import pytest
from scipy.io import wavfile


def _write_split_zip(
    directory,
    stem: str,
    members: dict[str, bytes],
    part_size: int | None = None,
    zip64: bool = False,
    n_parts: int | None = None,
):
    """Write a split archive the way Info-ZIP does: per-part offsets, split marker, EOCD on the last part.

    Give either ``part_size`` or ``n_parts`` (total number of parts, >= 2; the part size is then fitted).
    With ``zip64=True`` sizes/offsets are saturated in the central directory and stored in ZIP64 extra
    fields, and a ZIP64 end-of-central-directory record + locator are written.
    """
    if (part_size is None) == (n_parts is None):
        raise ValueError("Pass exactly one of part_size or n_parts.")
    stream = bytearray(b"PK\x07\x08")
    central = []
    for name, data in members.items():
        raw = name.encode("utf-8")
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        deflated = comp.compress(data) + comp.flush()
        method, payload = (8, deflated) if len(deflated) < len(data) else (0, data)
        crc = zlib.crc32(data)
        offset = len(stream)
        stream += struct.pack(
            "<4sHHHHHIIIHH", b"PK\x03\x04", 20, 0x800, method, 0, 0, crc, len(payload), len(data), len(raw), 0
        )
        stream += raw + payload
        central.append((raw, method, crc, len(payload), len(data), offset))

    if n_parts is not None:
        part_size = -(-len(stream) // (n_parts - 1))  # data fills n_parts - 1 parts; the CD gets the last

    def locate(offset):
        return offset // part_size, offset % part_size

    cd = bytearray()
    for raw, method, crc, csize, usize, offset in central:
        disk, rel = locate(offset)
        extra = b""
        if zip64:
            extra = struct.pack("<HHQQQI", 0x0001, 28, usize, csize, rel, disk)
            csize = usize = rel = 0xFFFFFFFF
            disk = 0xFFFF
        cd += struct.pack(
            "<4sHHHHHHIIIHHHHHII",
            b"PK\x01\x02",
            20,
            20,
            0x800,
            method,
            0,
            0,
            crc,
            csize,
            usize,
            len(raw),
            len(extra),
            0,
            disk,
            0,
            0,
            rel,
        )
        cd += raw + extra

    # The central directory and end records must start on the last part, as Info-ZIP writes them.
    n_body_parts = -(-len(stream) // part_size)
    stream += b"\0" * (n_body_parts * part_size - len(stream))  # pad so the CD starts a fresh part
    last_disk = n_body_parts
    cd_rel = len(stream) - last_disk * part_size
    stream += cd
    if zip64:
        eocd64_rel = len(stream) - last_disk * part_size
        stream += struct.pack(
            "<4sQHHIIQQQQ",
            b"PK\x06\x06",
            44,
            45,
            45,
            last_disk,
            last_disk,
            len(central),
            len(central),
            len(cd),
            cd_rel,
        )
        stream += struct.pack("<4sIQI", b"PK\x06\x07", last_disk, eocd64_rel, last_disk + 1)
        stream += struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    else:
        stream += struct.pack(
            "<4sHHHHIIH", b"PK\x05\x06", last_disk, last_disk, len(central), len(central), len(cd), cd_rel, 0
        )

    paths = []
    for disk in range(last_disk + 1):
        chunk = stream[disk * part_size : (disk + 1) * part_size] if disk < last_disk else stream[disk * part_size :]
        path = directory / (f"{stem}.z{disk + 1:02d}" if disk < last_disk else f"{stem}.zip")
        path.write_bytes(bytes(chunk))
        paths.append(path)
    return paths


def _wav_bytes(samples, sample_rate: int = 44100) -> bytes:
    """Encode int16 samples (``[T]`` mono or ``[T, C]``) as a PCM WAV."""
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, np.asarray(samples, dtype=np.int16))
    return buf.getvalue()


def _write_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


@pytest.fixture
def write_split_zip():
    """``write_split_zip(directory, stem, members, part_size=None, zip64=False, n_parts=None) -> [part paths]``."""
    return _write_split_zip


@pytest.fixture
def wav_bytes():
    """``wav_bytes(samples, sample_rate=44100) -> bytes`` of a PCM16 WAV."""
    return _wav_bytes


@pytest.fixture
def write_zip():
    """``write_zip(path, {member: bytes}) -> path`` of a deflated ZIP."""
    return _write_zip


@pytest.fixture
def fake_downloads(monkeypatch):
    """Patch a builder module's ``bulk_download`` to return local files keyed by asset filename."""

    def install(module: str, files_by_name: dict):
        def _fake_bulk_download(infos, dest_folder, checksums=None):
            return [files_by_name[info.filename] for info in infos]

        monkeypatch.setattr(f"{module}.bulk_download", _fake_bulk_download)

    return install


class HTTPRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file over HTTP range requests, with a read-ahead buffer.

    Lets the real-data tests open the multi-GB Zenodo archives in place: reading a ZIP's central
    directory and a few members transfers megabytes instead of the whole file.
    """

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.bytes_fetched = url, block, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
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
            with urllib.request.urlopen(request, timeout=60) as response:
                self._buf = response.read()
            self._buf_start = self.pos
            self.bytes_fetched += len(self._buf)
        offset = self.pos - self._buf_start
        self.pos += n
        return self._buf[offset : offset + n]


@pytest.fixture
def remote_file():
    """``remote_file(url) -> HTTPRangeFile``: open a remote file for random access without downloading it."""
    return HTTPRangeFile
