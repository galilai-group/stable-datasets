import pytest

from stable_datasets.video.action100m import Action100M


class TestAction100MMetadata:
    """Test Action100M class metadata."""

    def test_info_returns_dict(self):
        """Test that info() returns expected metadata."""
        info = Action100M.info()
        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "Action100M"
        assert "homepage" in info
        assert "huggingface_repo" in info
        assert "license" in info
        assert "citation" in info

    def test_class_attributes(self):
        """Test class-level attributes are set."""
        assert Action100M.HOMEPAGE is not None
        assert Action100M.HUGGINGFACE_REPO == "facebook/Action100M-preview"
        assert Action100M.LICENSE == "FAIR Noncommercial Research License"

    def test_get_video_segments_field_names(self):
        """Test that get_video_segments uses correct official field names."""
        # Mock sample with official schema field names
        sample = {
            "video_uid": "test123",
            "nodes": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "node_id": 1,
                    "parent_id": 0,
                    "level": 1,
                    "plm_caption": "A person walks",
                    "plm_action": "walking",
                    "llama3_caption": "Someone is walking down the street",
                    "gpt": {
                        "brief_summary": "Person walking",
                        "detailed_summary": "A person walks down a city street",
                        "brief_action": "walk",
                        "detailed_action": "walking down the street",
                        "actor": "person",
                    },
                }
            ],
        }

        segments = Action100M.get_video_segments(sample)
        assert len(segments) == 1
        seg = segments[0]

        # Check official field names are used
        assert "plm_caption" in seg
        assert seg["plm_caption"] == "A person walks"
        assert "plm_action" in seg
        assert seg["plm_action"] == "walking"
        assert "llama3_caption" in seg
        assert seg["llama3_caption"] == "Someone is walking down the street"
        assert "gpt" in seg
        assert isinstance(seg["gpt"], dict)

        # Verify old incorrect field names are NOT present
        assert "caption_plm" not in seg
        assert "caption_llama" not in seg
        assert "caption_gpt" not in seg

    def test_get_gpt_annotations(self):
        """Test GPT annotation extraction helper."""
        segment_with_gpt = {
            "start": 0.0,
            "end": 5.0,
            "gpt": {
                "brief_summary": "Person walking",
                "detailed_action": "walking down the street",
                "actor": "person",
            },
        }
        segment_without_gpt = {"start": 0.0, "end": 5.0}
        segment_empty_gpt = {"start": 0.0, "end": 5.0, "gpt": None}

        gpt = Action100M.get_gpt_annotations(segment_with_gpt)
        assert gpt is not None
        assert gpt["brief_summary"] == "Person walking"
        assert gpt["actor"] == "person"

        assert Action100M.get_gpt_annotations(segment_without_gpt) is None
        assert Action100M.get_gpt_annotations(segment_empty_gpt) is None


@pytest.mark.large
class TestAction100MStreaming:
    """Tests that require network access to HuggingFace."""

    def test_streaming_load(self):
        """Test loading dataset in streaming mode."""
        ds = Action100M(streaming=True, split="train")

        # Get first sample
        sample = next(iter(ds))

        # Check expected fields exist
        assert "video_uid" in sample
        assert isinstance(sample["video_uid"], str)

    def test_iter_samples_with_limit(self):
        """Test iter_samples helper with max_samples."""
        ds = Action100M(streaming=True, split="train")

        samples = list(Action100M.iter_samples(ds, max_samples=5))
        assert len(samples) == 5

        for sample in samples:
            assert "video_uid" in sample

    def test_get_video_segments(self):
        """Test segment extraction from sample."""
        ds = Action100M(streaming=True, split="train")
        sample = next(iter(ds))

        segments = Action100M.get_video_segments(sample)

        # segments may be empty for some samples, but should be a list
        assert isinstance(segments, list)

    def test_real_data_field_names(self):
        """Validate real dataset uses expected field names in nodes."""
        ds = Action100M(streaming=True, split="train")

        # Check multiple samples to find one with populated nodes
        valid_caption_fields = {"plm_caption", "plm_action", "llama3_caption", "gpt"}
        invalid_old_fields = {"caption_plm", "caption_llama", "caption_gpt"}

        found_caption_field = False
        for sample in Action100M.iter_samples(ds, max_samples=10):
            nodes = sample.get("nodes", [])
            for node in nodes:
                node_keys = set(node.keys())

                # Verify old incorrect field names are NOT present
                assert not (node_keys & invalid_old_fields), (
                    f"Found old field names in real data: {node_keys & invalid_old_fields}"
                )

                # Check if any valid caption field exists
                if node_keys & valid_caption_fields:
                    found_caption_field = True

        # At least one sample should have caption fields
        assert found_caption_field, "No caption fields found in first 10 samples"

    def test_gpt_field_is_dict(self):
        """Validate gpt field in real data is a dict, not a string."""
        ds = Action100M(streaming=True, split="train")

        for sample in Action100M.iter_samples(ds, max_samples=20):
            nodes = sample.get("nodes", [])
            for node in nodes:
                if "gpt" in node and node["gpt"] is not None:
                    assert isinstance(node["gpt"], dict), f"Expected gpt to be dict, got {type(node['gpt'])}"
                    return  # Found and validated

        pytest.skip("No gpt field found in first 20 samples")
