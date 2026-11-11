"""JPEG frame stores: lance datasets of JPEG frames (one per row), converted losslessly so any subset of a frame's
MCUs can be read and decoded, and read fast from RAM, local disk or network storage.

    from stable_datasets.framestore import convert_store, FrameStore, SharedClipLoader

    convert_store("/vol/frames.lance")                            # in place, resumable, pixel-identical (checked)
    store = FrameStore("/vol/frames.lance", index="shared")       # zero-copy reads; per-node index
    loader = SharedClipLoader("/vol/frames.lance", batch=64)      # decoded clips on the GPU

Modules:
    codec        header probes, lossless restart re-coding (C, csrc/jpeg_restart.c), MCU segment lengths
    decoders     whole-frame decoders: nvJPEG, libjpeg-turbo (and the NVJPG engines, for verification)
    convert      the converter (python -m stable_datasets.framestore.convert)
    store        FrameStore: blob descriptors, read policies (mmap, pread, spans, ranges, take), clip windows
    mp4          Mp4Store: one H.264 MP4 per video, every clip from a keyframe, decoded straight to pixels
    index        FrameIndex: a per-node index of every frame's location and MCU segment lengths
    preads       many preads per call from C threads
    resources    the share of a node a job may use, and holding it to that
    pixel_loader SharedClipLoader: whole-frame decode into a GPU ring served by MPS processes
"""

__all__ = [
    "ConvertError",
    "Decoder",
    "FrameIndex",
    "FrameStore",
    "Mp4Store",
    "Resources",
    "SharedClipLoader",
    "convert_store",
]

_WHERE = {
    "convert_store": "convert",
    "ConvertError": "convert",
    "Decoder": "decoders",
    "FrameIndex": "index",
    "FrameStore": "store",
    "Mp4Store": "mp4",
    "Resources": "resources",
    "SharedClipLoader": "pixel_loader",
}


def __getattr__(name):  # lazy: importing the package does not import torch or lance
    if name not in _WHERE:
        raise AttributeError(f"module 'stable_datasets.framestore' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
