"""The Storage interface and the registry of cache layouts.

This module imports nothing from stable_datasets at import time: features/video.py imports VideoStorage, so
the storage modules must load before the schema does. A layout module loads when the registry first needs it.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from ..cache import CacheOpenResult, LanceCacheMeta
    from ..dataset import StableDataset
    from ..schema import Features


_BUILTIN = {
    "lance-rows": "stable_datasets.storage.rows:RowStorage",
    "lance-video-frames": "stable_datasets.storage.video_frames:VideoFramesStorage",
    "framestore-video": "stable_datasets.storage.video:VideoStorage",
}
_LAYOUTS: dict[str, type[Storage]] = {}


@dataclass(frozen=True)
class Storage:
    """How a split is written to its cache directory and opened again.

    A subclass is a frozen dataclass of the settings that change what is written. ``layout`` names the layout in
    the cache metadata; ``open_cache`` uses it to find the class again. ``aliases`` are the strings
    ``Storage.coerce`` accepts for the class, each with the settings it stands for.
    """

    layout: ClassVar[str]
    aliases: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    # Order in which storage_of asks the layouts; the row layout answers last and always.
    priority: ClassVar[int] = 10

    @classmethod
    def from_features(cls, features: Features) -> Storage | None:
        """The storage that features declare (e.g. ``Video(storage="frames")``), or None."""
        return None

    def apply(self, features: Features) -> Features:
        """Features changed to declare this storage. Storage that features declare is part of the cache key."""
        return features

    def write(
        self,
        generator,
        features: Features,
        cache_dir: str,
        *,
        lineage: dict | None = None,
        storage_options: dict | None = None,
    ) -> LanceCacheMeta:
        """Consume ``(key, example)`` pairs and write the cache, its ``_metadata.json`` last."""
        raise NotImplementedError

    @classmethod
    def open(
        cls,
        cache_dir: str,
        features: Features,
        metadata: dict,
        *,
        backend_kwargs: dict,
        storage_options: dict | None = None,
    ) -> CacheOpenResult:
        """The backend of a cache this layout wrote. ``metadata`` is its ``_metadata.json``."""
        raise NotImplementedError

    def loader(self, dataset: StableDataset, batch_size: int, **kwargs):
        """A batch loader over a dataset in this storage."""
        raise NotImplementedError(f"{type(self).__name__} has no batch loader")

    @staticmethod
    def coerce(value) -> Storage:
        """A Storage from itself or from an alias of a layout, e.g. ``"mp4"`` or ``"frames"``."""
        if isinstance(value, Storage):
            return value
        if isinstance(value, str):
            for cls in layouts():
                if value in cls.aliases:
                    return cls(**cls.aliases[value])
            names = sorted(a for cls in layouts() for a in cls.aliases)
            raise ValueError(f"Unknown storage {value!r}. Known: {names}")
        raise TypeError(f"storage must be a Storage or one of its aliases, got {type(value).__name__}")


def register_layout(cls: type[Storage]) -> type[Storage]:
    """Register a Storage class under its ``layout`` name (a class decorator)."""
    name = cls.__dict__.get("layout")
    if not isinstance(name, str):
        raise TypeError(f"{cls.__name__} must set the class attribute 'layout'")
    known = _LAYOUTS.get(name)
    if known is not None and known is not cls:
        raise ValueError(f"layout {name!r} is already registered by {known.__name__}")
    _LAYOUTS[name] = cls
    return cls


def _load_builtins() -> None:
    for name, target in _BUILTIN.items():
        if name not in _LAYOUTS:
            module, attr = target.split(":")
            getattr(importlib.import_module(module), attr)


def layout_class(name: str) -> type[Storage]:
    """The Storage class registered for a layout name."""
    if name not in _LAYOUTS and name in _BUILTIN:
        module, attr = _BUILTIN[name].split(":")
        getattr(importlib.import_module(module), attr)
    if name not in _LAYOUTS:
        raise KeyError(name)
    return _LAYOUTS[name]


def layouts() -> list[type[Storage]]:
    """Every registered Storage class, highest priority first."""
    _load_builtins()
    return sorted(_LAYOUTS.values(), key=lambda cls: -cls.priority)


def storage_of(features: Features) -> Storage:
    """The storage of a dataset with these features."""
    for cls in layouts():
        storage = cls.from_features(features)
        if storage is not None:
            return storage
    raise ValueError("no storage layout accepts these features")
