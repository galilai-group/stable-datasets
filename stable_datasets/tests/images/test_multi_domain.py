"""Tests for the helpers shared by PACS and Office-Home (``images/_multi_domain.py``)."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images._multi_domain import (
    ensure_rgb,
    google_drive_url,
    image_members,
    normalize_name,
    num_test_examples,
    stratified_test_keys,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Real World", "real_world"), ("Art_Painting", "art_painting"), (" clip-art ", "clip_art"), ("photo", "photo")],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_google_drive_url():
    assert google_drive_url("abc").endswith("id=abc&export=download&confirm=t")
    assert google_drive_url("abc", "0-key").endswith("&resourcekey=0-key")


@pytest.mark.parametrize(("n", "expected"), [(0, 0), (1, 0), (2, 1), (4, 1), (5, 1), (8, 2), (10, 2), (100, 20)])
def test_num_test_examples(n, expected):
    assert num_test_examples(n, 0.2) == expected


def test_num_test_examples_keeps_both_sides_nonempty():
    for n in range(2, 200):
        assert 1 <= num_test_examples(n, 0.2) <= n - 1


def test_stratified_test_keys_is_per_group_and_order_independent():
    items = [(f"{g}-{i}", f"{g}-{i}".encode(), (g,)) for g in range(3) for i in range(10)]
    keys = stratified_test_keys(items, 0.2)
    assert len(keys) == 6
    assert {k.split("-")[0] for k in keys} == {"0", "1", "2"}
    assert stratified_test_keys(list(reversed(items)), 0.2) == keys


def test_image_members_matches_domain_folders(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Top/Real World/Mug/1.jpg", b"x")
        z.writestr("Top/Art/Mug/2.PNG", b"x")
        z.writestr("Top/ImageInfo.csv", b"x")
        z.writestr("Top/Other/Mug/3.jpg", b"x")
    with zipfile.ZipFile(path) as z:
        assert image_members(z, ["art", "real_world"]) == {
            "real_world/Mug/1.jpg": "Top/Real World/Mug/1.jpg",
            "art/Mug/2.PNG": "Top/Art/Mug/2.PNG",
        }


def _encode(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_ensure_rgb_keeps_rgb_bytes():
    data = _encode(Image.fromarray(np.zeros((4, 4, 3), np.uint8)), "JPEG")
    assert ensure_rgb(data) is data


@pytest.mark.parametrize("mode", ["L", "P", "CMYK"])
def test_ensure_rgb_converts(mode):
    img = Image.fromarray(np.full((4, 4, 3), 100, np.uint8)).convert(mode)
    out = Image.open(io.BytesIO(ensure_rgb(_encode(img, "JPEG" if mode == "CMYK" else "PNG"))))
    assert out.mode == "RGB"


def test_ensure_rgb_puts_transparency_on_white():
    rgba = np.zeros((4, 4, 4), np.uint8)  # fully transparent black
    out = Image.open(io.BytesIO(ensure_rgb(_encode(Image.fromarray(rgba, "RGBA")))))
    assert out.mode == "RGB"
    assert np.asarray(out).min() == 255
