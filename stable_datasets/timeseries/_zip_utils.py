"""Read members of single or split (multi-part) ZIP archives.

Some datasets (e.g. FSD50K on Zenodo) ship their audio as a *split* ZIP:
``name.z01, name.z02, ..., name.zip``. Python's :mod:`zipfile` refuses these
("zipfiles that span multiple disks are not supported"), and merging the parts
with ``zip -s 0`` first would need tens of GB of extra disk space.

:class:`SplitZipFile` reads members directly from the parts. A split archive is
laid out as if the parts were concatenated, except that every offset stored in
the archive (local header offsets and the central-directory offset) is relative
to the start of the part ("disk") it lives on. We therefore map each
``(disk, offset)`` pair to a global offset into the virtual concatenation of the
parts and read from there; members may straddle part boundaries.

Supported: stored and deflated members, ZIP64 records. Not supported: encrypted
members and other compression methods (they raise ``NotImplementedError``).
"""

from __future__ import annotations

import bisect
import io
import re
import struct
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOCATOR_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
_CENTRAL_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_SPLIT_MARKER = b"PK\x07\x08"

_EOCD_SIZE = 22
_EOCD64_LOCATOR_SIZE = 20
_MAX_COMMENT = 0xFFFF
_ZIP64_EXTRA_ID = 0x0001


@dataclass(frozen=True)
class SplitZipInfo:
    """Central-directory entry of a member in a (split) ZIP archive."""

    filename: str
    compress_type: int
    compress_size: int
    file_size: int
    crc: int
    flag_bits: int
    disk: int
    header_offset: int

    def is_dir(self) -> bool:
        return self.filename.endswith("/")


class _ConcatReader:
    """Random-access reads over an ordered list of seekable binary streams."""

    def __init__(self, streams: Sequence[BinaryIO]):
        self._streams = list(streams)
        self.sizes = [s.seek(0, io.SEEK_END) for s in self._streams]
        self.starts = [0]
        for size in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + size)
        self.total = self.starts[-1] + self.sizes[-1]

    def global_offset(self, disk: int, offset: int) -> int:
        if not 0 <= disk < len(self._streams):
            raise zipfile.BadZipFile(f"Archive references part {disk + 1}, but only {len(self._streams)} parts given.")
        return self.starts[disk] + offset

    def read_at(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.total:
            raise zipfile.BadZipFile(f"Read of {size} bytes at offset {offset} is outside the archive.")
        chunks = []
        part = bisect.bisect_right(self.starts, offset) - 1
        while size > 0:
            local = offset - self.starts[part]
            n = min(size, self.sizes[part] - local)
            stream = self._streams[part]
            stream.seek(local)
            chunk = stream.read(n)
            if len(chunk) != n:
                raise zipfile.BadZipFile(f"Unexpected end of part {part + 1}.")
            chunks.append(chunk)
            offset += n
            size -= n
            part += 1
        return b"".join(chunks)


class SplitZipFile:
    """Read-only access to a split ZIP archive given its parts in order.

    Args:
        parts: The archive parts in disk order, i.e. ``[name.z01, ..., name.zNN, name.zip]``.
            Items may be paths or seekable binary file objects (the latter are not closed by
            :meth:`close`). Use :func:`split_zip_parts` to discover the parts of a ``.zip`` file.

    Example:
        >>> with SplitZipFile(split_zip_parts("FSD50K.eval_audio.zip")) as archive:  # doctest: +SKIP
        ...     wav_bytes = archive.read("FSD50K.eval_audio/37199.wav")
    """

    def __init__(self, parts: Sequence[str | Path | BinaryIO]):
        if not parts:
            raise ValueError("SplitZipFile needs at least one part.")
        self._owned: list[BinaryIO] = []
        streams = []
        for part in parts:
            if isinstance(part, str | Path):
                stream = open(part, "rb")  # noqa: SIM115 - closed in close()
                self._owned.append(stream)
            else:
                stream = part
            streams.append(stream)
        try:
            self._reader = _ConcatReader(streams)
            self._infos = self._read_central_directory()
        except BaseException:
            self.close()
            raise
        self._by_name = {info.filename: info for info in self._infos}

    # -- public API (mirrors the subset of zipfile.ZipFile that builders use) --

    def infolist(self) -> list[SplitZipInfo]:
        return list(self._infos)

    def namelist(self) -> list[str]:
        return [info.filename for info in self._infos]

    def getinfo(self, name: str) -> SplitZipInfo:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"There is no item named {name!r} in the archive") from None

    def read(self, name: str | SplitZipInfo) -> bytes:
        """Return the decompressed bytes of a member, verifying its CRC-32."""
        info = name if isinstance(name, SplitZipInfo) else self.getinfo(name)
        if info.flag_bits & 0x1:
            raise NotImplementedError(f"{info.filename!r} is encrypted; encrypted members are not supported.")

        start = self._reader.global_offset(info.disk, info.header_offset)
        header = self._reader.read_at(start, 30)
        if header[:4] != _LOCAL_SIG:
            raise zipfile.BadZipFile(f"Bad local file header for {info.filename!r}.")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        raw = self._reader.read_at(start + 30 + name_len + extra_len, info.compress_size)

        if info.compress_type == zipfile.ZIP_STORED:
            data = raw
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(raw, -15)
        else:
            raise NotImplementedError(
                f"Compression method {info.compress_type} of {info.filename!r} is not supported."
            )

        if len(data) != info.file_size or zlib.crc32(data) != info.crc:
            raise zipfile.BadZipFile(f"CRC or size mismatch for {info.filename!r}; the archive may be corrupt.")
        return data

    def open(self, name: str | SplitZipInfo) -> io.BytesIO:
        """Return a member as an in-memory binary stream."""
        return io.BytesIO(self.read(name))

    def close(self) -> None:
        for stream in self._owned:
            stream.close()
        self._owned = []

    def __enter__(self) -> SplitZipFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- parsing --

    def _read_central_directory(self) -> list[SplitZipInfo]:
        reader = self._reader
        last = len(reader.sizes) - 1
        tail_len = min(reader.sizes[last], _EOCD_SIZE + _MAX_COMMENT)
        tail_start = reader.total - tail_len
        tail = reader.read_at(tail_start, tail_len)
        pos = tail.rfind(_EOCD_SIG)
        if pos < 0:
            raise zipfile.BadZipFile("End of central directory record not found in the last part.")

        (this_disk, cd_disk, _, n_entries, cd_size, cd_offset) = struct.unpack("<HHHHII", tail[pos + 4 : pos + 20])
        if 0xFFFF in (this_disk, cd_disk, n_entries) or 0xFFFFFFFF in (cd_size, cd_offset):
            this_disk, cd_disk, n_entries, cd_size, cd_offset = self._read_eocd64(tail_start + pos)

        if this_disk != last:
            raise zipfile.BadZipFile(
                f"Archive has {this_disk + 1} parts but {last + 1} were given; a part is missing or extra."
            )
        if last > 0 and reader.read_at(0, 4) != _SPLIT_MARKER:
            raise zipfile.BadZipFile("First part does not start with the split-archive marker; check the part order.")

        cd = reader.read_at(reader.global_offset(cd_disk, cd_offset), cd_size)
        return _parse_central_directory(cd, n_entries)

    def _read_eocd64(self, eocd_global: int) -> tuple[int, int, int, int, int]:
        locator = self._reader.read_at(eocd_global - _EOCD64_LOCATOR_SIZE, _EOCD64_LOCATOR_SIZE)
        if locator[:4] != _EOCD64_LOCATOR_SIG:
            raise zipfile.BadZipFile("ZIP64 end of central directory locator not found.")
        eocd64_disk, eocd64_offset, _ = struct.unpack("<IQI", locator[4:])
        record = self._reader.read_at(self._reader.global_offset(eocd64_disk, eocd64_offset), 56)
        if record[:4] != _EOCD64_SIG:
            raise zipfile.BadZipFile("ZIP64 end of central directory record not found.")
        this_disk, cd_disk, _, n_entries, cd_size, cd_offset = struct.unpack("<IIQQQQ", record[16:56])
        return this_disk, cd_disk, n_entries, cd_size, cd_offset


def _parse_central_directory(cd: bytes, n_entries: int) -> list[SplitZipInfo]:
    infos = []
    pos = 0
    for _ in range(n_entries):
        if cd[pos : pos + 4] != _CENTRAL_SIG:
            raise zipfile.BadZipFile("Bad central directory entry.")
        # Fixed 46-byte header; skip signature + "version made by" + "version needed" (8 bytes),
        # mod time/date (4) and internal/external attributes (6).
        (flag_bits, method, crc, csize, usize, name_len, extra_len, comment_len, disk, offset) = struct.unpack(
            "<HH4xIIIHHHH6xI", cd[pos + 8 : pos + 46]
        )
        name_raw = cd[pos + 46 : pos + 46 + name_len]
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        name = name_raw.decode("utf-8" if flag_bits & 0x800 else "cp437")
        usize, csize, offset, disk = _apply_zip64_extra(extra, usize, csize, offset, disk)
        infos.append(SplitZipInfo(name, method, csize, usize, crc, flag_bits, disk, offset))
        pos += 46 + name_len + extra_len + comment_len
    return infos


def _apply_zip64_extra(extra: bytes, usize: int, csize: int, offset: int, disk: int) -> tuple[int, int, int, int]:
    """Replace saturated 32/16-bit fields with their ZIP64 extra-field values (spec order)."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[pos : pos + 4])
        if header_id == _ZIP64_EXTRA_ID:
            data = extra[pos + 4 : pos + 4 + size]
            fields = iter(struct.unpack(f"<{len(data) // 8}Q", data[: len(data) // 8 * 8]))
            if usize == 0xFFFFFFFF:
                usize = next(fields)
            if csize == 0xFFFFFFFF:
                csize = next(fields)
            if offset == 0xFFFFFFFF:
                offset = next(fields)
            if disk == 0xFFFF:
                disk = struct.unpack("<I", data[-4:])[0]
            break
        pos += 4 + size
    return usize, csize, offset, disk


def split_zip_parts(zip_path: str | Path) -> list[Path]:
    """Return ``[name.z01, ..., name.zNN, name.zip]`` for ``name.zip`` (just ``[name.zip]`` if unsplit)."""
    zip_path = Path(zip_path)
    pattern = re.compile(re.escape(zip_path.stem) + r"\.z(\d{2,})$", re.IGNORECASE)
    numbered = sorted(
        (int(m.group(1)), p) for p in zip_path.parent.iterdir() if (m := pattern.match(p.name)) is not None
    )
    expected = list(range(1, len(numbered) + 1))
    if [n for n, _ in numbered] != expected:
        raise FileNotFoundError(
            f"Split archive parts of {zip_path.name} are not contiguous: {[p.name for _, p in numbered]}"
        )
    return [p for _, p in numbered] + [zip_path]


def open_zip(parts: str | Path | Sequence[str | Path]) -> zipfile.ZipFile | SplitZipFile:
    """Open a single ZIP with :mod:`zipfile`, or a split ZIP (several parts) with :class:`SplitZipFile`.

    Both returned objects support ``namelist()``, ``read(name)``, ``open(name)`` and use as a
    context manager, which is all the dataset builders need.
    """
    if isinstance(parts, str | Path):
        parts = [parts]
    if len(parts) == 1:
        return zipfile.ZipFile(parts[0])
    return SplitZipFile(parts)
