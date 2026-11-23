"""Storage layouts: how a split is written to its cache and read back.

    from stable_datasets.storage import VideoStorage
    ds = SomethingSomethingV2(split="train", storage=VideoStorage(fps=7.5, short_edge=256))

Layouts: ``RowStorage`` (one Lance row per example, the default), ``VideoFramesStorage`` (one WebP frame per
row) and ``VideoStorage`` (a framestore store of MP4s or JPEG frames). A new layout subclasses ``Storage`` and
registers with ``register_layout``.
"""

import importlib


_WHERE = {
    "Storage": "base",
    "layout_class": "base",
    "layouts": "base",
    "register_layout": "base",
    "storage_of": "base",
    "RowStorage": "rows",
    "VideoFramesStorage": "video_frames",
    "VideoStorage": "video",
}

__all__ = list(_WHERE)


def __getattr__(name):  # lazy: the schema imports VideoStorage before the layout modules can load
    if name not in _WHERE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{_WHERE[name]}"), name)
