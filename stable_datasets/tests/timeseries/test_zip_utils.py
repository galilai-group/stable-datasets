import os
import shutil
import subprocess
import zipfile

import pytest

from stable_datasets.timeseries._zip_utils import SplitZipFile, open_zip, split_zip_parts


def _payloads() -> dict[str, bytes]:
    rng = __import__("random").Random(0)
    return {
        "audio/a.wav": bytes(rng.getrandbits(8) for _ in range(3000)),  # incompressible
        "audio/b.wav": b"tone" * 2500,  # compressible
        "audio/c.wav": b"",
        "audio/ünï.wav": bytes(rng.getrandbits(8) for _ in range(1777)),
    }


@pytest.mark.parametrize("zip64", [False, True])
def test_split_zip_roundtrip_including_members_straddling_parts(tmp_path, write_split_zip, zip64):
    members = _payloads()
    parts = write_split_zip(tmp_path, "arch", members, part_size=1024, zip64=zip64)
    assert len(parts) >= 4

    with SplitZipFile(parts) as archive:
        assert archive.namelist() == list(members)
        for name, data in members.items():
            assert archive.read(name) == data
            assert archive.open(name).read() == data
        disks = {info.disk for info in archive.infolist()}
    assert len(disks) > 1  # members really live on different parts


def test_split_zip_parts_discovers_parts_in_numeric_order(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    (tmp_path / "other.z01").write_bytes(b"x")  # unrelated archive in the same folder
    assert split_zip_parts(tmp_path / "arch.zip") == parts


def test_split_zip_parts_of_unsplit_archive(tmp_path):
    path = tmp_path / "single.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x.txt", "hi")
    assert split_zip_parts(path) == [path]


def test_split_zip_parts_rejects_gaps(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    parts[1].unlink()
    with pytest.raises(FileNotFoundError, match="not contiguous"):
        split_zip_parts(tmp_path / "arch.zip")


def test_missing_part_is_reported(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    with pytest.raises(zipfile.BadZipFile, match="part is missing"):
        SplitZipFile(parts[1:])


def test_parts_out_of_order_are_reported(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    with pytest.raises(zipfile.BadZipFile, match="part order"):
        SplitZipFile([parts[1], parts[0], *parts[2:]])


def test_corrupt_member_fails_crc(tmp_path, write_split_zip):
    members = {"a.bin": os.urandom(4000)}
    parts = write_split_zip(tmp_path, "arch", members, part_size=1024)
    corrupted = bytearray(parts[1].read_bytes())
    corrupted[10] ^= 0xFF
    parts[1].write_bytes(bytes(corrupted))
    with SplitZipFile(parts) as archive, pytest.raises(zipfile.BadZipFile, match="CRC"):
        archive.read("a.bin")


def test_unknown_member_raises_keyerror(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    with SplitZipFile(parts) as archive, pytest.raises(KeyError):
        archive.read("nope.wav")


def test_open_zip_dispatches_on_number_of_parts(tmp_path, write_split_zip):
    single = tmp_path / "single.zip"
    with zipfile.ZipFile(single, "w") as archive:
        archive.writestr("x.txt", "hi")
    with open_zip(single) as archive:
        assert isinstance(archive, zipfile.ZipFile)
        assert archive.read("x.txt") == b"hi"

    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    with open_zip(parts) as archive:
        assert isinstance(archive, SplitZipFile)
        assert archive.read("audio/b.wav") == b"tone" * 2500


def test_close_releases_owned_files(tmp_path, write_split_zip):
    parts = write_split_zip(tmp_path, "arch", _payloads(), part_size=512)
    archive = SplitZipFile(parts)
    archive.close()
    assert archive._owned == []


@pytest.mark.skipif(shutil.which("zip") is None, reason="Info-ZIP `zip` not installed")
@pytest.mark.parametrize("force_zip64", [False, True])
def test_reads_archives_split_by_info_zip(tmp_path, force_zip64):
    src = tmp_path / "src"
    (src / "audio").mkdir(parents=True)
    members = {f"audio/{i}.wav": os.urandom(20_000 + 3_000 * i) for i in range(5)}
    for name, data in members.items():
        (src / name).write_bytes(data)
    cmd = ["zip", "-q", "-r", "-s", "64k", *(["-fz"] if force_zip64 else []), str(tmp_path / "arch.zip"), "audio"]
    subprocess.run(cmd, cwd=src, check=True)

    parts = split_zip_parts(tmp_path / "arch.zip")
    assert len(parts) > 1
    with SplitZipFile(parts) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        assert {info.filename for info in files} == set(members)
        for info in files:
            assert archive.read(info) == members[info.filename]
