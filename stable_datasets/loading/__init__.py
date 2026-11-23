"""Loading: sampling, plans, and their execution.

A batch goes sampler -> transform draws -> regions -> codec units -> byte ranges -> coalesced reads -> decode ->
apply. Each step's output is arrays, decided before any byte is read, so reads can be merged, planned ahead, and
served from any ByteSource.
"""

from .engine import PlanLoader, build_batch
from .plan import Plan, Planner, Records, decode_sample, fetch, fetch_and_decode
from .sampling import ClassBalanced, Sampler, SamplingIndex, Uniform, Weighted, clip_windows


__all__ = [
    "ClassBalanced",
    "Plan",
    "PlanLoader",
    "Planner",
    "Records",
    "Sampler",
    "SamplingIndex",
    "Uniform",
    "Weighted",
    "build_batch",
    "clip_windows",
    "decode_sample",
    "fetch",
    "fetch_and_decode",
]
