import json
from unittest.mock import MagicMock, patch

import pytest

from stable_datasets.video.vpt_minecraft import VPTMinecraft


class TestVPTMinecraftMetadata:
    """Test VPTMinecraft class metadata (no network required)."""

    def test_info_returns_dict(self):
        """Test that info() returns expected metadata."""
        info = VPTMinecraft.info()
        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "VPTMinecraft"
        assert "homepage" in info
        assert "license" in info
        assert "citation" in info
        assert "versions" in info
        assert "description" in info

    def test_class_attributes(self):
        """Test class-level attributes are set correctly."""
        assert VPTMinecraft.HOMEPAGE == "https://github.com/openai/Video-Pre-Training"
        assert VPTMinecraft.LICENSE == "MIT"
        assert VPTMinecraft.CITATION is not None
        assert "baker2022video" in VPTMinecraft.CITATION

    def test_available_versions(self):
        """Test that all expected versions are present in INDEX_URLS."""
        expected = {"v6", "v7", "v8", "v9", "v10", "find_cave", "waterfall", "animal_pen", "build_house"}
        assert expected == set(VPTMinecraft.INDEX_URLS.keys())

    def test_info_versions_match_index_urls(self):
        """Test that info()['versions'] matches INDEX_URLS keys."""
        info = VPTMinecraft.info()
        assert set(info["versions"]) == set(VPTMinecraft.INDEX_URLS.keys())

    def test_unknown_version_raises_value_error(self):
        """Test that an unrecognised version raises ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown version"):
            VPTMinecraft(version="v99_fake")

    def test_unknown_version_error_lists_available(self):
        """Test that the ValueError message includes available versions."""
        with pytest.raises(ValueError) as exc_info:
            VPTMinecraft(version="bad_version")
        assert "v10" in str(exc_info.value)
        assert "find_cave" in str(exc_info.value)

    def test_parse_relpath_standard(self):
        """Test parse_relpath with a standard well-formed relpath."""
        info = VPTMinecraft.parse_relpath("10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639")
        assert info["version"] == "10.0"
        assert info["contractor_alias"] == "cheeky-cornflower-setter"
        assert info["session_id"] == "02e496ce4abb"
        assert info["date"] == "20220421"
        assert info["time"] == "092639"

    def test_parse_relpath_version_prefix(self):
        """Test parse_relpath with a v8.x relpath."""
        info = VPTMinecraft.parse_relpath("8.0/cheeky-cornflower-setter-74ae6c2eae2e-20220315-122354")
        assert info["version"] == "8.0"
        assert info["date"] == "20220315"
        assert info["time"] == "122354"

    def test_iter_segments_with_limit(self):
        """Test iter_segments returns at most max_segments items."""
        fake = [{"relpath": f"10.0/seg-{i}"} for i in range(10)]
        result = list(VPTMinecraft.iter_segments(fake, max_segments=3))
        assert len(result) == 3
        assert result[0]["relpath"] == "10.0/seg-0"
        assert result[2]["relpath"] == "10.0/seg-2"

    def test_iter_segments_no_limit(self):
        """Test iter_segments with no limit yields all segments."""
        fake = [{"relpath": f"10.0/seg-{i}"} for i in range(7)]
        result = list(VPTMinecraft.iter_segments(fake))
        assert len(result) == 7

    def test_iter_segments_zero_limit(self):
        """Test iter_segments with max_segments=0 yields nothing."""
        fake = [{"relpath": "10.0/seg-0"}]
        result = list(VPTMinecraft.iter_segments(fake, max_segments=0))
        assert result == []

    def test_new_builds_correct_urls(self):
        """Test that __new__ constructs segment URLs correctly from index JSON."""
        mock_index = {
            "basedir": "https://openaipublic.blob.core.windows.net/minecraft-rl/data",
            "relpaths": [
                "10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639",
            ],
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_index
        mock_response.raise_for_status = MagicMock()

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            segments = VPTMinecraft(version="v10")

        assert len(segments) == 1
        seg = segments[0]
        base = "https://openaipublic.blob.core.windows.net/minecraft-rl/data"
        rp = "10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639"
        assert seg["relpath"] == rp
        assert seg["video_url"] == f"{base}/{rp}.mp4"
        assert seg["action_url"] == f"{base}/{rp}.jsonl"
        assert seg["options_url"] == f"{base}/{rp}-options.json"
        assert seg["checkpoint_url"] == f"{base}/{rp}.zip"

    def test_new_strips_trailing_slash_from_basedir(self):
        """Test that a trailing slash in basedir does not produce double slashes."""
        mock_index = {
            "basedir": "https://example.com/data/",
            "relpaths": ["10.0/test-segment-aabbccddeeff-20220101-010101"],
        }
        mock_response = MagicMock()
        mock_response.json.return_value = mock_index
        mock_response.raise_for_status = MagicMock()

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            segments = VPTMinecraft(version="v10")

        assert "//" not in segments[0]["video_url"].replace("https://", "")

    def test_load_actions_parses_jsonl(self):
        """Test load_actions correctly parses a JSONL response."""
        action1 = {
            "mouse": {"dx": 0.0, "dy": 0.0, "buttons": [], "newButtons": []},
            "keyboard": {"keys": ["key.keyboard.w"], "newKeys": [], "chars": ""},
            "isGuiOpen": False,
            "hotbar": 0,
            "yaw": -90.0,
            "pitch": 0.0,
            "tick": 0,
        }
        action2 = {
            "mouse": {"dx": 1.0, "dy": 0.5, "buttons": [], "newButtons": []},
            "keyboard": {"keys": [], "newKeys": [], "chars": ""},
            "isGuiOpen": False,
            "hotbar": 1,
            "yaw": -91.0,
            "pitch": 0.5,
            "tick": 1,
        }
        jsonl_body = json.dumps(action1) + "\n" + json.dumps(action2) + "\n"

        mock_response = MagicMock()
        mock_response.text = jsonl_body
        mock_response.raise_for_status = MagicMock()

        seg = {"action_url": "https://example.com/test.jsonl"}
        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            actions = VPTMinecraft.load_actions(seg)

        assert len(actions) == 2
        assert actions[0]["keyboard"]["keys"] == ["key.keyboard.w"]
        assert actions[0]["hotbar"] == 0
        assert actions[1]["mouse"]["dx"] == 1.0
        assert actions[1]["tick"] == 1

    def test_load_actions_skips_blank_lines(self):
        """Test that blank lines in the JSONL are ignored."""
        action = {"tick": 0, "keyboard": {"keys": []}}
        jsonl_body = "\n" + json.dumps(action) + "\n\n"

        mock_response = MagicMock()
        mock_response.text = jsonl_body
        mock_response.raise_for_status = MagicMock()

        seg = {"action_url": "https://example.com/test.jsonl"}
        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            actions = VPTMinecraft.load_actions(seg)

        assert len(actions) == 1
        assert actions[0]["tick"] == 0


class TestVPTMinecraftDownload:
    """Unit tests for download_segment / download_segments (no real network)."""

    def test_download_segment_unknown_file_type_raises(self):
        """Test that requesting an unknown file type raises ValueError."""
        seg = {
            "relpath": "10.0/test-aabbccddeeff-20220101-010101",
            "video_url": "https://example.com/test.mp4",
            "action_url": "https://example.com/test.jsonl",
            "options_url": "https://example.com/test-options.json",
            "checkpoint_url": "https://example.com/test.zip",
        }
        with pytest.raises(ValueError, match="Unknown file types"):
            VPTMinecraft.download_segment(seg, files=("video", "nonexistent"))

    def test_download_segment_skips_existing_file(self, tmp_path):
        """Test that an already-downloaded file is not re-fetched."""
        seg = {
            "relpath": "10.0/alias-aabbccddeeff-20220101-010101",
            "video_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.mp4",
            "action_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.jsonl",
            "options_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101-options.json",
            "checkpoint_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.zip",
        }
        # Pre-create the expected destination file
        dest = tmp_path / "vpt_minecraft" / "10.0" / "alias-aabbccddeeff-20220101-010101.jsonl"
        dest.parent.mkdir(parents=True)
        dest.write_text("existing content")

        with patch("stable_datasets.video.vpt_minecraft.requests.get") as mock_get:
            paths = VPTMinecraft.download_segment(seg, dest_dir=tmp_path, files=("actions",))

        mock_get.assert_not_called()
        assert paths["actions"] == dest

    def test_download_segment_writes_file(self, tmp_path):
        """Test that a missing file is downloaded and written correctly."""
        seg = {
            "relpath": "10.0/alias-aabbccddeeff-20220101-010101",
            "video_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.mp4",
            "action_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.jsonl",
            "options_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101-options.json",
            "checkpoint_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.zip",
        }
        content = b'{"tick": 0}\n{"tick": 1}\n'
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            paths = VPTMinecraft.download_segment(seg, dest_dir=tmp_path, files=("actions",), progress_bar=False)

        assert "actions" in paths
        assert paths["actions"].exists()
        assert paths["actions"].read_bytes() == content

    def test_download_segment_cleans_up_tmp_on_failure(self, tmp_path):
        """Test that the .tmp file is removed if a download fails."""
        seg = {
            "relpath": "10.0/alias-aabbccddeeff-20220101-010101",
            "video_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.mp4",
            "action_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.jsonl",
            "options_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101-options.json",
            "checkpoint_url": "https://example.com/10.0/alias-aabbccddeeff-20220101-010101.zip",
        }
        import requests as req

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("404")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            with pytest.raises(req.HTTPError):
                VPTMinecraft.download_segment(seg, dest_dir=tmp_path, files=("actions",), progress_bar=False)

        # No leftover .tmp files
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == [], f"Leftover tmp files: {tmp_files}"

    def test_download_segment_returns_correct_path_structure(self, tmp_path):
        """Test that downloaded files land under vpt_minecraft/{version}/."""
        seg = {
            "relpath": "8.0/builder-aabbccddeeff-20220301-120000",
            "video_url": "https://example.com/8.0/builder-aabbccddeeff-20220301-120000.mp4",
            "action_url": "https://example.com/8.0/builder-aabbccddeeff-20220301-120000.jsonl",
            "options_url": "https://example.com/8.0/builder-aabbccddeeff-20220301-120000-options.json",
            "checkpoint_url": "https://example.com/8.0/builder-aabbccddeeff-20220301-120000.zip",
        }
        content = b"fake video bytes"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            paths = VPTMinecraft.download_segment(seg, dest_dir=tmp_path, files=("video",), progress_bar=False)

        assert paths["video"].parent == tmp_path / "vpt_minecraft" / "8.0"

    def test_download_segments_returns_one_result_per_segment(self, tmp_path):
        """Test that download_segments returns a result for each input segment."""
        fake_segments = [
            {
                "relpath": f"10.0/alias-{'a' * 12}-2022010{i}-010101",
                "video_url": f"https://example.com/10.0/seg{i}.mp4",
                "action_url": f"https://example.com/10.0/seg{i}.jsonl",
                "options_url": f"https://example.com/10.0/seg{i}-options.json",
                "checkpoint_url": f"https://example.com/10.0/seg{i}.zip",
            }
            for i in range(3)
        ]
        content = b"data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            results = VPTMinecraft.download_segments(
                fake_segments,
                dest_dir=tmp_path,
                files=("actions",),
                max_workers=2,
            )

        assert len(results) == 3
        for r in results:
            assert "actions" in r

    def test_download_segments_respects_max_segments(self, tmp_path):
        """Test that max_segments caps the number of segments downloaded."""
        fake_segments = [
            {
                "relpath": f"10.0/alias-{'a' * 12}-2022010{i}-010101",
                "video_url": f"https://example.com/10.0/seg{i}.mp4",
                "action_url": f"https://example.com/10.0/seg{i}.jsonl",
                "options_url": f"https://example.com/10.0/seg{i}-options.json",
                "checkpoint_url": f"https://example.com/10.0/seg{i}.zip",
            }
            for i in range(5)
        ]
        content = b"data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("stable_datasets.video.vpt_minecraft.requests.get", return_value=mock_response):
            results = VPTMinecraft.download_segments(
                fake_segments,
                dest_dir=tmp_path,
                files=("actions",),
                max_segments=2,
            )

        assert len(results) == 2


@pytest.mark.large
class TestVPTMinecraftNetwork:
    """Tests that require live network access to the OpenAI public mirror."""

    def test_load_v10_index(self):
        """Test fetching the v10 index returns a non-empty segment list."""
        segments = VPTMinecraft(version="v10")
        assert isinstance(segments, list)
        assert len(segments) > 0

    def test_segment_has_expected_keys(self):
        """Test that every segment dict contains the required keys."""
        segments = VPTMinecraft(version="v10")
        required_keys = {"relpath", "video_url", "action_url", "options_url", "checkpoint_url"}
        for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
            assert required_keys.issubset(seg.keys()), f"Missing keys in segment: {required_keys - seg.keys()}"

    def test_video_urls_end_with_mp4(self):
        """Test that video_url fields have the .mp4 extension."""
        segments = VPTMinecraft(version="v10")
        for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
            assert seg["video_url"].endswith(".mp4"), seg["video_url"]

    def test_action_urls_end_with_jsonl(self):
        """Test that action_url fields have the .jsonl extension."""
        segments = VPTMinecraft(version="v10")
        for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
            assert seg["action_url"].endswith(".jsonl"), seg["action_url"]

    def test_relpaths_contain_version(self):
        """Test that relpaths carry a recorder version prefix."""
        segments = VPTMinecraft(version="v10")
        for seg in VPTMinecraft.iter_segments(segments, max_segments=5):
            assert "/" in seg["relpath"], f"Unexpected relpath format: {seg['relpath']}"

    def test_load_actions_returns_action_dicts(self):
        """Test fetching actions from the known IDM demo segment."""
        seg = {
            "action_url": (
                "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/"
                "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl"
            )
        }
        actions = VPTMinecraft.load_actions(seg)
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_action_dict_has_mouse_and_keyboard(self):
        """Test that each action dict from real data has mouse and keyboard fields."""
        seg = {
            "action_url": (
                "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/"
                "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl"
            )
        }
        actions = VPTMinecraft.load_actions(seg)
        for action in actions[:10]:
            assert "mouse" in action, "Expected 'mouse' key in action dict"
            assert "keyboard" in action, "Expected 'keyboard' key in action dict"

    def test_all_versions_return_segments(self):
        """Test that all non-BASALT recorder versions return at least one segment."""
        for version in ("v6", "v7", "v8", "v9", "v10"):
            segments = VPTMinecraft(version=version)
            assert len(segments) > 0, f"Version {version} returned no segments"

    def test_basalt_versions_return_segments(self):
        """Test that all BASALT versions return at least one segment."""
        for version in ("find_cave", "waterfall", "animal_pen", "build_house"):
            segments = VPTMinecraft(version=version)
            assert len(segments) > 0, f"BASALT version {version} returned no segments"

    def test_fetch_video_chunk_returns_bytes(self):
        """Test that fetch_video_chunk downloads a small slice of a real video."""
        seg = {
            "video_url": (
                "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/"
                "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"
            )
        }
        chunk = VPTMinecraft.fetch_video_chunk(seg, num_bytes=4096)
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0
        assert len(chunk) <= 4096

    def test_fetch_video_chunk_is_mp4(self):
        """Test that the returned chunk begins with a valid MP4/ftyp box header."""
        seg = {
            "video_url": (
                "https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/"
                "cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4"
            )
        }
        chunk = VPTMinecraft.fetch_video_chunk(seg, num_bytes=4096)
        # All MP4 files start with a 4-byte box size followed by b'ftyp'
        assert b"ftyp" in chunk[:12], f"Expected MP4 ftyp box in first 12 bytes, got: {chunk[:12]!r}"
