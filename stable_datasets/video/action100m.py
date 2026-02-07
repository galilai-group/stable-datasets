"""Action100M dataset wrapper for stable-datasets.

Action100M is a large-scale video action dataset containing ~100M YouTube videos
with hierarchical "Tree-of-Captions" annotations. Due to its massive scale, this
dataset is designed for streaming access.

The dataset provides:
- video_uid: YouTube video identifier
- metadata: Video-level information (title, description, ASR transcript)
- nodes: Segment-level annotations with temporal boundaries and multiple caption types

Note: This dataset does NOT download actual videos. It provides YouTube video IDs
and annotations. Users must fetch videos separately using tools like yt-dlp.

License: FAIR Noncommercial Research License
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from datasets import IterableDataset, load_dataset


if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDatasetDict

# Dataset metadata
HOMEPAGE = "https://github.com/facebookresearch/Action100M"
HUGGINGFACE_REPO = "facebook/Action100M-preview"
CITATION = """@misc{action100m2024,
    title={Action100M: A Large-Scale Video Action Dataset},
    author={Facebook AI Research},
    year={2024},
    howpublished={\\url{https://github.com/facebookresearch/Action100M}}
}"""
LICENSE = "FAIR Noncommercial Research License"


class Action100M:
    """Action100M dataset: Large-scale video action annotations.

    This class provides a convenient interface to the Action100M dataset hosted
    on HuggingFace. Due to the dataset's massive scale (~100M videos), streaming
    mode is enabled by default.

    The dataset contains hierarchical "Tree-of-Captions" annotations with:
    - Multiple temporal granularities
    - Brief and detailed action descriptions
    - Imperative instructions
    - Actor information

    Examples:
        # Streaming mode (recommended for large-scale processing)
        >>> ds = Action100M(streaming=True)
        >>> for sample in ds:
        ...     print(sample["video_uid"])
        ...     break

        # Non-streaming mode (downloads parquet files locally)
        >>> ds = Action100M(streaming=False)

        # Access specific split
        >>> ds = Action100M(streaming=True, split="train")

    Note:
        This dataset only provides video UIDs and annotations. Actual video
        files must be downloaded separately using the YouTube video IDs.
    """

    # Class-level metadata for consistency with BaseDatasetBuilder pattern
    HOMEPAGE = HOMEPAGE
    HUGGINGFACE_REPO = HUGGINGFACE_REPO
    CITATION = CITATION
    LICENSE = LICENSE

    def __new__(
        cls,
        streaming: bool = True,
        split: str | None = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> Dataset | DatasetDict | IterableDataset | IterableDatasetDict:
        """Load the Action100M dataset.

        Args:
            streaming: If True (default), streams data without downloading.
                Recommended for this large dataset.
            split: Dataset split to load. If None, loads all available splits.
            trust_remote_code: Whether to trust remote code from HuggingFace.
            **kwargs: Additional arguments passed to datasets.load_dataset().

        Returns:
            The loaded dataset. Type depends on streaming and split parameters:
            - streaming=True, split=None: IterableDatasetDict
            - streaming=True, split="train": IterableDataset
            - streaming=False, split=None: DatasetDict
            - streaming=False, split="train": Dataset
        """
        # Load from HuggingFace using parquet files
        dataset = load_dataset(
            "parquet",
            data_files=f"hf://datasets/{HUGGINGFACE_REPO}/data/*.parquet",
            streaming=streaming,
            split=split,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return dataset

    @classmethod
    def info(cls) -> dict[str, str]:
        """Return dataset metadata."""
        return {
            "name": "Action100M",
            "homepage": cls.HOMEPAGE,
            "huggingface_repo": cls.HUGGINGFACE_REPO,
            "license": cls.LICENSE,
            "citation": cls.CITATION,
            "description": (
                "Large-scale video action dataset with ~100M YouTube videos "
                "and hierarchical Tree-of-Captions annotations."
            ),
        }

    @staticmethod
    def iter_samples(
        dataset: IterableDataset,
        max_samples: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over samples with optional limit.

        Convenience method for iterating over streaming datasets.

        Args:
            dataset: An IterableDataset from Action100M.
            max_samples: Maximum number of samples to yield. None for unlimited.

        Yields:
            Sample dictionaries from the dataset.

        Example:
            >>> ds = Action100M(streaming=True, split="train")
            >>> for sample in Action100M.iter_samples(ds, max_samples=10):
            ...     print(sample["video_uid"])
        """
        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            yield sample

    @staticmethod
    def get_video_segments(sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract temporal segments from a sample's nodes.

        Args:
            sample: A sample dictionary from the dataset.

        Returns:
            List of segment dictionaries with temporal boundaries and captions.
            Caption fields use the official schema names:
            - plm_caption: PLM-generated caption
            - plm_action: PLM-generated action description
            - llama3_caption: LLaMA3-generated caption
            - gpt: GPT annotations dict (use get_gpt_annotations() to extract)

        Example:
            >>> ds = Action100M(streaming=True, split="train")
            >>> sample = next(iter(ds))
            >>> segments = Action100M.get_video_segments(sample)
            >>> for seg in segments[:3]:
            ...     print(f"{seg['start']}-{seg['end']}: {seg.get('plm_caption', 'N/A')}")
        """
        nodes = sample.get("nodes", [])
        if not nodes:
            return []

        segments = []
        for node in nodes:
            segment = {
                "start": node.get("start"),
                "end": node.get("end"),
                "node_id": node.get("node_id"),
                "parent_id": node.get("parent_id"),
                "level": node.get("level"),
            }
            # Extract available captions using official schema field names
            for caption_key in ["plm_caption", "plm_action", "llama3_caption"]:
                if caption_key in node:
                    segment[caption_key] = node[caption_key]
            # Handle gpt dict separately (contains structured annotations)
            if "gpt" in node and node["gpt"]:
                segment["gpt"] = node["gpt"]
            segments.append(segment)

        return segments

    @staticmethod
    def get_gpt_annotations(segment: dict[str, Any]) -> dict[str, Any] | None:
        """Extract GPT annotations from a segment.

        The GPT field contains structured annotations with brief/detailed
        summaries, action descriptions, and actor information.

        Args:
            segment: A segment dictionary (from get_video_segments()).

        Returns:
            Dict with GPT annotations if present, None otherwise.
            Expected keys in the returned dict:
            - brief_summary: Brief summary of the segment
            - detailed_summary: Detailed summary of the segment
            - brief_action: Brief action description
            - detailed_action: Detailed action description
            - actor: Actor information

        Example:
            >>> segments = Action100M.get_video_segments(sample)
            >>> for seg in segments:
            ...     gpt = Action100M.get_gpt_annotations(seg)
            ...     if gpt:
            ...         print(gpt.get("brief_action"))
        """
        gpt = segment.get("gpt")
        if not gpt:
            return None
        return gpt

    @staticmethod
    def get_youtube_url(video_uid: str) -> str:
        """Construct YouTube URL from video UID.

        Args:
            video_uid: YouTube video identifier.

        Returns:
            Full YouTube URL for the video.

        Example:
            >>> url = Action100M.get_youtube_url("dQw4w9WgXcQ")
            >>> print(url)
            https://www.youtube.com/watch?v=dQw4w9WgXcQ
        """
        return f"https://www.youtube.com/watch?v={video_uid}"
