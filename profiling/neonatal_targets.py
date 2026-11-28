"""Published neonatal annotations and an explicit diagnostic window policy.

The MAT file contains one (three experts, seconds) array per subject. Labels
are aligned to the start of the EDF. This experiment uses majority agreement
per second and marks a window positive if any of its seconds is positive.
That is an experiment policy, not a clinical evaluation standard.
"""

from pathlib import Path

import numpy as np


def load_expert_labels(path: Path, subject_id: int) -> np.ndarray:
    """Read binary expert labels without changing their one-second time base."""
    from scipy.io import loadmat

    cells = loadmat(path)["annotat_new"]
    if cells.shape != (1, 79) or not isinstance(subject_id, int) or not 1 <= subject_id <= 79:
        raise ValueError("Require annotat_new shaped (1, 79) and a subject ID from 1 to 79")
    values = np.asarray(cells[0, subject_id - 1])
    if values.ndim != 2 or values.shape[0] != 3 or not values.shape[1] or not np.isin(values, [0, 1]).all():
        raise ValueError("Require three experts with binary per-second annotations")
    return values.astype(np.uint8)


def window_targets(experts: np.ndarray, starts: np.ndarray, length: int, rate_hz: float) -> np.ndarray:
    """Label whole-second windows; reject ambiguous subsecond boundaries."""
    if (
        experts.ndim != 2
        or experts.shape[0] != 3
        or not np.isin(experts, [0, 1]).all()
        or not np.isfinite(rate_hz)
        or rate_hz < 1
        or rate_hz != int(rate_hz)
    ):
        raise ValueError("Require three binary experts and an integral sample rate")
    rate = int(rate_hz)
    if (
        not np.issubdtype(starts.dtype, np.integer)
        or np.any(starts < 0)
        or np.any(starts % rate)
        or length < rate
        or length % rate
        or np.any(starts // rate + length // rate > experts.shape[1])
    ):
        raise ValueError("Windows must use whole annotated seconds within the recording")
    positive = experts.sum(axis=0) >= 2
    cumulative = np.concatenate(([0], np.cumsum(positive)))
    begin = starts // rate
    return (cumulative[begin + length // rate] > cumulative[begin]).astype(np.int64)
