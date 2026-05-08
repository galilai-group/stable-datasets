"""Split name constants and split generator."""

from __future__ import annotations

from dataclasses import dataclass, field


class Split:
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


@dataclass
class SplitGenerator:
    """Describes one split and the kwargs to pass to ``_generate_examples``."""

    name: str
    gen_kwargs: dict = field(default_factory=dict)
