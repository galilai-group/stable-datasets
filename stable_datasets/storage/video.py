"""The framestore video layout (``framestore-video``): the video column built as a stable_datasets.framestore store.

The schema imports VideoStorage, so this module imports only the standard library at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import Storage, register_layout


if TYPE_CHECKING:
    from ..cache import CacheOpenResult, LanceCacheMeta
    from ..dataset import StableDataset
    from ..schema import Features


@register_layout
@dataclass(frozen=True)
class VideoStorage(Storage):
    """How a dataset stores its videos when it is built (stable_datasets.framestore), for fast clip reads.

    ``format="mp4"`` (the default) keeps one H.264 MP4 per video, re-encoded so that every ``keyint``-th frame is a
    keyframe a clip can start on with no earlier frame decoded, every video coded at one size (``canvas``; portrait
    videos transposed, readers return each video as it was) and decodable on the GPU (framestore.gpu_h264) as well as
    by ffmpeg or NVDEC. ``format="jpeg"`` keeps every frame as a JPEG with a restart marker after each MCU, so any
    part of a frame decodes alone (partial decoding: the token loader).

    Which to choose, measured on an H100 with Kinetics-400 at 7.5 fps, 384 px short edge (the whole-frame loader
    alone, 224 x 224 clips of 16 frames; framestore's benchmarks): MP4 stores are ~half the size of JPEG ones at
    the same frame rate (about the size of the source MP4s) and load 4,131 clips/s (GPU and CPU decoding) against
    3,369 for JPEG; JPEG stores allow partial decoding, any frame stride when loading, and decode faster on the CPU
    alone. A size check never switches the format: it is what is asked for.

    fps, short_edge: frames kept per second (the clip stride is fixed then, for MP4) and the short edge in pixels
    (None: the source's). crf, keyint, canvas: MP4's libx264 quality, frames between keyframes, coded size ("auto":
    16:9 at the short edge; (H, W); None: each video's own). quality: JPEG's.
    """

    layout = "framestore-video"
    aliases = {"mp4": {"format": "mp4"}, "jpeg": {"format": "jpeg"}}

    format: str = "mp4"
    fps: float | None = None
    short_edge: int | None = None
    crf: int = 18
    keyint: int = 4
    canvas: str | tuple[int, int] | None = "auto"
    quality: int = 90

    def __post_init__(self):
        if self.format not in ("mp4", "jpeg"):
            raise ValueError(f"VideoStorage.format must be 'mp4' or 'jpeg', got {self.format!r}.")
        if self.keyint < 1 or self.crf < 0:
            raise ValueError("VideoStorage needs keyint >= 1 and crf >= 0.")
        if isinstance(self.canvas, list):
            object.__setattr__(self, "canvas", tuple(self.canvas))

    @classmethod
    def from_features(cls, features: Features) -> VideoStorage | None:
        from ..features.video import Video

        return next(
            (f.settings for f in features.values() if isinstance(f, Video) and f.storage in Video.STORES), None
        )

    def apply(self, features: Features) -> Features:
        from ..features.video import Video
        from ..schema import Features

        return Features(
            (name, Video(storage=self.format, allowed_extensions=f.allowed_extensions, settings=self))
            if isinstance(f, Video) else (name, f)
            for name, f in features.items()
        )  # fmt: skip

    def write(self, generator, features, cache_dir, *, lineage=None, storage_options=None) -> LanceCacheMeta:
        from ._video_store import write_framestore_video_cache

        return write_framestore_video_cache(
            generator, features, cache_dir, lineage=lineage, storage_options=storage_options
        )

    @classmethod
    def open(cls, cache_dir, features, metadata, *, backend_kwargs, storage_options=None) -> CacheOpenResult:
        from ..backends.framestore_video import FramestoreVideoBackend
        from ..cache import CacheOpenResult, _check_format_version, _check_schema_fingerprint

        backend = FramestoreVideoBackend(uri=cache_dir, **{"storage_options": storage_options, **backend_kwargs})
        _check_format_version(backend.metadata)
        _check_schema_fingerprint(backend.metadata["schema_fingerprint"], features, cache_dir)
        return CacheOpenResult(
            backend=backend, num_rows=backend.num_rows, layout=cls.layout, metadata=backend.metadata
        )

    def loader(
        self,
        dataset: StableDataset,
        batch_size: int,
        *,
        size=(224, 224),
        decoders: str | None = None,
        seed: int = 0,
        **kwargs,
    ):
        """A stable_datasets.framestore.SharedClipLoader over the store: (batch_size, T, H, W, 3) uint8 clips on
        the GPU, drawn iid (a video uniformly, then a window start in it), T and the frame stride from the
        dataset's video decode (set_video_decode(num_frames=..., frame_stride=...)).

        decoders: framestore's decoder spec ("triton:2,cpu": two GPU decode servers plus the CPU; MP4 stores also
        take "nvdec"); None: "triton:2,cpu". kwargs go to SharedClipLoader. The loader decodes whole frames and
        resizes them to size. Batches hold the frames only: read the other fields (labels) from the dataset."""
        from ..features.video import Video
        from ..framestore import SharedClipLoader

        features = dataset.features
        config = dataset._video_decode_config
        col = (
            config.column
            if config is not None
            else next((n for n, f in features.items() if isinstance(f, Video) and f.storage in Video.STORES), None)
        )
        feat = features.get(col)
        if not isinstance(feat, Video) or feat.storage not in Video.STORES:
            raise ValueError("loader() needs a video column stored as a framestore store (storage='mp4' or 'jpeg')")
        frames = config.num_frames if config is not None else 16
        stride = config.frame_stride if config is not None else 1
        return SharedClipLoader(
            dataset._backend.store_path, batch_size, frames=frames, stride=stride, size=tuple(size),
            decoders=decoders or "triton:2,cpu", seed=seed, **kwargs,
        )  # fmt: skip
