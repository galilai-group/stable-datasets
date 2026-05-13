"""Utilities for the independent per-method RankMe analysis.

This module intentionally does not import ``benchmarks.analysis``.  It fetches
W&B runs, extracts final scalar metrics, and selects the highest-probe run per
``(model, dataset)`` while carrying the matching kNN and RankMe values from
that same run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
import requests


DEFAULT_ENTITY = "samibg"
DEFAULT_PROJECT = "finalized-stable-datasets"
DEFAULT_BACKBONE = "vit_small_patch16_224"
DEFAULT_BACKBONES: tuple[str, ...] = (DEFAULT_BACKBONE, "vit_small")

SUPERVISED_METHOD = "supervised"
SSL_METHODS: tuple[str, ...] = ("simclr", "barlow_twins", "nnclr", "dino", "lejepa", "mae")
ALL_METHODS: tuple[str, ...] = (*SSL_METHODS, SUPERVISED_METHOD)

DATASET_ALIASES: dict[str, str] = {"medmnist": "pneumoniamnist"}

PROBE_KEYS: tuple[str, ...] = ("eval/linear_probe_top1_epoch",)
KNN_KEYS: tuple[str, ...] = ("eval/knn_probe_top1",)
RANKME_VAL_KEYS: tuple[str, ...] = ("rankme", "val/rankme", "eval/rankme", "rankme/val")


@dataclass(frozen=True)
class RunIdentity:
    """Normalized identity parsed from a W&B run config/name."""

    model: str
    backbone: str
    dataset: str
    seed: int | None = None


def retry(fn, max_retries: int = 5):
    """Call ``fn`` with exponential backoff for W&B/API rate limits."""
    for attempt in range(max_retries):
        try:
            return fn()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                time.sleep(2**attempt)
                continue
            raise
    return fn()


def _as_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_seed(dataset_part: str) -> tuple[str, int | None]:
    if "_seed" not in dataset_part:
        return dataset_part, None
    dataset, _, tail = dataset_part.rpartition("_seed")
    try:
        return dataset, int(tail)
    except ValueError:
        return dataset_part, None


def normalize_backbones(
    backbones: Sequence[str] | str | None = None,
    *,
    backbone: str | None = None,
) -> tuple[str, ...]:
    """Normalize backbone CLI/API inputs while preserving preference order."""
    if backbones is None:
        raw = (backbone,) if backbone is not None else DEFAULT_BACKBONES
    elif isinstance(backbones, str):
        raw = tuple(part.strip() for part in backbones.split(","))
    else:
        raw = tuple(backbones)

    out: list[str] = []
    for b in raw:
        if b and b not in out:
            out.append(b)
    if not out:
        raise ValueError("at least one backbone is required")
    return tuple(out)


def parse_run_name(
    name: str | None,
    *,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
    methods: Iterable[str] = ALL_METHODS,
) -> RunIdentity | None:
    """Parse ``{model}_{backbone}_{dataset}[_seedN]`` from a W&B run name."""
    if not name:
        return None
    method_set = set(methods)
    for backbone in sorted(backbones, key=len, reverse=True):
        tag = f"_{backbone}_"
        if tag not in name:
            continue
        model, _, dataset_part = name.partition(tag)
        if model not in method_set:
            continue
        dataset, seed = _parse_seed(dataset_part.lower())
        dataset = DATASET_ALIASES.get(dataset, dataset)
        return RunIdentity(model=model, backbone=backbone, dataset=dataset, seed=seed)
    return None


def _model_from_config(config: dict) -> str | None:
    model = config.get("model") or config.get("model/name")
    if isinstance(model, dict):
        model = model.get("name")
    return str(model) if model else None


def _seed_from_config(config: dict) -> int | None:
    seed = config.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def identity_from_run(
    run,
    *,
    backbones: Sequence[str] | str = DEFAULT_BACKBONES,
    methods: Iterable[str] = ALL_METHODS,
) -> RunIdentity | None:
    """Resolve model/backbone/dataset from W&B config, falling back to name."""
    allowed_backbones = normalize_backbones(backbones)
    method_set = set(methods)
    config = retry(lambda: run.config or {})
    parsed = parse_run_name(run.name, backbones=allowed_backbones, methods=method_set)

    model = _model_from_config(config) or (parsed.model if parsed else None)
    dataset = (config.get("dataset") or (parsed.dataset if parsed else None) or "").lower()
    config_backbone = str(config.get("backbone") or "")
    run_backbone = parsed.backbone if parsed is not None else config_backbone
    seed = _seed_from_config(config)
    if seed is None and parsed is not None:
        seed = parsed.seed

    dataset = DATASET_ALIASES.get(dataset, dataset)
    if not model or model not in method_set or not dataset:
        return None
    if not run_backbone or run_backbone not in allowed_backbones:
        return None
    return RunIdentity(model=model, backbone=run_backbone, dataset=dataset, seed=seed)


def summary_dict(run) -> dict:
    summary = retry(lambda: run.summary)
    if hasattr(summary, "_json_dict"):
        return dict(summary._json_dict)
    return dict(summary)


def _last_history_value(run, key: str, samples: int) -> float | None:
    try:
        hist = retry(lambda: run.history(keys=[key], samples=samples, pandas=True))
    except Exception:
        return None
    if hist is None or hist.empty or key not in hist.columns:
        return None
    values = hist[key].dropna()
    if values.empty:
        return None
    return _as_scalar(values.iloc[-1])


def final_metric(
    run,
    summary: dict,
    keys: Sequence[str],
    *,
    history_samples: int = 0,
) -> tuple[float | None, str | None]:
    """Return the final scalar for the first available W&B key."""
    for key in keys:
        val = _as_scalar(summary.get(key))
        if val is not None:
            return val, key
    if history_samples <= 0:
        return None, None
    for key in keys:
        val = _last_history_value(run, key, samples=history_samples)
        if val is not None:
            return val, key
    return None, None


def collect_run_metrics(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    *,
    backbone: str | None = None,
    backbones: Sequence[str] | str | None = None,
    methods: Iterable[str] = ALL_METHODS,
    datasets: Iterable[str] | None = None,
    history_samples: int = 0,
) -> pd.DataFrame:
    """Fetch W&B run metrics and keep only runs with any RankMe metric."""
    import wandb

    allowed_backbones = normalize_backbones(backbones, backbone=backbone)
    method_set = set(methods)
    dataset_set = {DATASET_ALIASES.get(d.lower(), d.lower()) for d in datasets} if datasets else None

    api = wandb.Api(timeout=60)
    runs = retry(lambda: list(api.runs(f"{entity}/{project}", per_page=1000)))

    rows: list[dict] = []
    for run in runs:
        state = retry(lambda: run.state)
        if state not in {"finished", "completed"}:
            continue

        identity = identity_from_run(run, backbones=allowed_backbones, methods=method_set)
        if identity is None:
            continue
        if dataset_set is not None and identity.dataset not in dataset_set:
            continue

        summary = summary_dict(run)
        probe, probe_key = final_metric(run, summary, PROBE_KEYS, history_samples=history_samples)
        knn, knn_key = final_metric(run, summary, KNN_KEYS, history_samples=history_samples)
        rankme_val, rankme_val_key = final_metric(run, summary, RANKME_VAL_KEYS, history_samples=history_samples)

        if probe is None or rankme_val is None:
            continue

        rows.append(
            {
                "model": identity.model,
                "backbone": identity.backbone,
                "dataset": identity.dataset,
                "seed": identity.seed,
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "probe": probe,
                "knn": knn,
                "rankme": rankme_val,
                "rankme_val": rankme_val,
                "probe_key": probe_key,
                "knn_key": knn_key,
                "rankme_val_key": rankme_val_key,
            }
        )

    return pd.DataFrame(rows)


def select_best_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Select highest-probe run per ``(model, dataset)`` with matching metrics."""
    if runs.empty:
        return runs.copy()
    required = {"model", "dataset", "probe"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs is missing required columns: {sorted(missing)}")
    return (
        runs.dropna(subset=["probe"])
        .sort_values(["model", "dataset", "probe"], ascending=[True, True, False])
        .drop_duplicates(subset=["model", "dataset"], keep="first")
        .reset_index(drop=True)
    )


def fetch_best_metrics(
    backbone: str,
    dataset: str,
    *,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    backbones: Sequence[str] | str | None = None,
    methods: Iterable[str] = ALL_METHODS,
    history_samples: int = 0,
) -> pd.DataFrame:
    """Fetch best W&B metrics for one ``(backbone, dataset)`` pair.

    Pass ``backbones=...`` to allow more than the positional backbone.
    """
    runs = collect_run_metrics(
        entity,
        project,
        backbone=backbone,
        backbones=backbones,
        methods=methods,
        datasets=[dataset],
        history_samples=history_samples,
    )
    return select_best_runs(runs)
