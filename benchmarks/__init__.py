"""Benchmark helpers; model dependencies load only when explicitly requested."""

from importlib import import_module


__all__ = ["dataset", "models"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"benchmarks.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
