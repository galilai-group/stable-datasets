"""Compute RankMe + LiDAR on trained benchmark checkpoints.

For each (model, dataset) checkpoint produced by benchmarks/run.py:

  1. Rebuild the Lightning module from the hydra config.
  2. Load the checkpoint's ``state_dict``.
  3. Forward the standard val split through the eval-time encoder, collecting
     ``(E, y)`` where E is the (N, d) embedding matrix and y the labels.
  4. Compute RankMe (label-free, SVD entropy) and LiDAR (labeled LDA trace-
     ratio entropy, delegating to the SPT implementation) on that (E, y).

Why not the SPT callbacks directly: RankMe/LiDAR in stable_pretraining are
training-time monitors — their underlying OnlineQueue only pushes on
``on_train_batch_end``, so a one-shot ``trainer.validate()`` never fills
the queue and the callback returns ``None``. For post-hoc evaluation on
finished checkpoints we extract embeddings ourselves and call SPT's
LiDAR math (``LiDAR._compute_lidar``) on a tensor grouped by real class
labels — more faithful to the paper than the callback's sequential
surrogate classes, since we have actual labels rather than q augmented
views of a clean sample.

Records go through ``ExperimentManager`` to
``benchmarks/results/representation_scores_history.json``, keyed on
(model, dataset, seed, split) so reruns are idempotent.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from stable_pretraining.callbacks.lidar import LiDAR as _LiDARCallback

from benchmarks.analysis.experiment_manager import ExperimentManager
from benchmarks.analysis.utils import (
    BACKBONE,
    COLLAPSED,
    CONFIG_DIR,
    HISTORY_PATHS,
    LIDAR_SAMPLES_PER_CLASS,
    MAX_SAMPLES,
    discover_all,
    latest_checkpoint,
    run_dir,
)
from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import build_module, collate_single, get_transforms, val_transform


def build_cfg(model: str, dataset: str):
    """Compose the hydra config that run.py would have built."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model}",
                f"backbone={BACKBONE}",
                f"dataset={dataset}",
                "wandb.enabled=false",
            ],
        )
    with open_dict(cfg):
        cfg.model._total_steps = 1  # build_module wants this for the scheduler
    return cfg


def resolve_encoder(module):
    """Return the eval-time encoder.

    * DINO / iBOT wrap the student in TeacherStudentWrapper; the teacher is
      the one used downstream. Expose ``.teacher`` if present.
    * LeJEPA nests the backbone one level deeper under ``module.model.backbone``
      (see benchmarks/models/lejepa.py:163).
    * Everything else exposes the plain ``module.backbone``.
    """
    try:
        backbone = module.backbone
    except AttributeError:
        backbone = module.model.backbone
    return backbone.teacher if hasattr(backbone, "teacher") else backbone


@torch.no_grad()
def extract_embeddings(encoder, loader, device, max_samples: int):
    """Forward val batches through the frozen encoder, return (E, y)."""
    embs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        out = encoder(images)
        if isinstance(out, torch.Tensor):
            feats = out[:, 0] if out.dim() == 3 else out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state[:, 0]
        elif hasattr(out, "encoded"):
            # MAE's MaskedEncoderOutput: (B, num_prefix + N_patches, D). Masking
            # is disabled under module.eval() (see stable_pretraining vit.py:194),
            # so encoded covers all patches. Take CLS — same pattern as
            # benchmarks/models/mae.py:51.
            feats = out.encoded[:, 0]
        else:
            raise RuntimeError(f"unrecognized encoder output type: {type(out)}")
        embs.append(feats.cpu().float())
        labels.append(y.cpu().long() if isinstance(y, torch.Tensor) else torch.tensor(y).long())
        n += feats.shape[0]
        if n >= max_samples:
            break
    E = torch.cat(embs, dim=0)[:max_samples]
    y = torch.cat(labels, dim=0)[:max_samples]
    return E, y


def rankme(E: torch.Tensor, eps: float = 1e-7) -> float:
    """Soft rank of the embedding matrix. Garrido et al. 2023.

    RankMe(E) = exp(-Σ p_i log p_i),  p_i = σ_i / Σ σ_j over singular values
    of E. Label-free. Identical to the SVD block in
    ``stable_pretraining.callbacks.rankme.RankMe.on_validation_batch_end``
    — inlined here because we extract features in a plain loop rather than
    through the training-time queue the callback expects.
    """
    with torch.no_grad():
        E = E.double()
        s = torch.linalg.svdvals(E)
        p = (s / s.sum()) + eps
        entropy = -(p * p.log()).sum()
        return float(torch.exp(entropy).item())


def _lidar_helper() -> _LiDARCallback:
    """Cheap, non-functional LiDAR callback we only use for its math kernel."""
    # target_shape is irrelevant here; we never call setup/queue.
    return _LiDARCallback(
        name="_postmortem_lidar",
        target="embedding",
        queue_length=1,
        target_shape=1,
    )


def lidar(E: torch.Tensor, y: torch.Tensor, samples_per_class: int = LIDAR_SAMPLES_PER_CLASS
          ) -> tuple[float | None, int, int]:
    """LDA-rank entropy using real class labels. Delegates to SPT's kernel.

    For each class with ≥ ``samples_per_class`` embeddings we sample exactly
    that many (without replacement, fixed seed). We stack into
    ``(n_classes, samples_per_class, d)`` and hand it to SPT's
    ``LiDAR._compute_lidar``, which runs the full LDA whitening + trace-
    ratio eigendecomposition + entropy-over-eigenvalues from the paper.
    Returns (lidar_value, n_classes_used, samples_per_class).
    """
    with torch.no_grad():
        E = E.double()
        E = F.normalize(E, dim=-1)
        classes, inv = torch.unique(y, return_inverse=True)

        rng = np.random.default_rng(0)
        grouped: list[torch.Tensor] = []
        for c_idx in range(classes.numel()):
            mask = (inv == c_idx).nonzero(as_tuple=True)[0]
            if mask.numel() < samples_per_class:
                continue
            pick = rng.choice(mask.numel(), size=samples_per_class, replace=False)
            grouped.append(E[mask[pick]])

        n_classes_used = len(grouped)
        if n_classes_used < 2:
            return None, n_classes_used, samples_per_class

        stacked = torch.stack(grouped, dim=0)  # (n_classes, samples_per_class, d)
        flat = stacked.view(n_classes_used * samples_per_class, -1)

        # Minimum input to _compute_lidar: a (n_samples, d) tensor that
        # reshapes to (n_classes, samples_per_class, d) via its internal view.
        # Ensure n_classes matches the callback's expectation via its param.
        helper = _lidar_helper()
        helper.n_classes = n_classes_used
        helper.samples_per_class = samples_per_class
        val = helper._compute_lidar(flat)
        return (float(val) if val is not None else None), n_classes_used, samples_per_class


def _build_loader(split: str, model: str, dataset: str, cfg, ds_config):
    """Build the appropriate DataLoader for a split.

    val: model's default transforms + collate (so the loader is the same shape
         the SSL run validated on).
    train: eval transform + ``collate_single`` so the train images are
           preprocessed identically to val (no augmentation noise) and the
           batch is single-view, not multi-view as SSL collates produce.
    """
    if split == "val":
        train_t, val_t, collate = get_transforms(model, ds_config)
    elif split == "train":
        vt = val_transform(ds_config)
        train_t, val_t, collate = vt, vt, collate_single
    else:
        raise ValueError(f"unknown split: {split}")
    data, _ = create_dataset(
        dataset, train_t, val_t, collate, cfg.training, data_dir=cfg.get("data_dir")
    )
    return data.train if split == "train" else data.val


def process_one(model: str, dataset: str, split: str, device: torch.device,
                mgr: ExperimentManager, overwrite: bool = False,
                dry_run: bool = False) -> dict | None:
    if (model, dataset) in COLLAPSED:
        print(f"[{model}/{dataset}] in COLLAPSED skip-set; skip", flush=True)
        return None

    if mgr.has(model, dataset, seed=None, split=split) and not overwrite:
        print(f"[{model}/{dataset}] already in {split} history; skip", flush=True)
        return None

    ckpt = latest_checkpoint(model, dataset)
    if ckpt is None:
        print(f"[{model}/{dataset}] no checkpoint at {run_dir(model, dataset)}; skip", flush=True)
        return None

    print(f"[{model}/{dataset}] ckpt = {ckpt.name}", flush=True)
    t_build = time.time()
    cfg = build_cfg(model, dataset)
    ds_config = get_config(dataset)
    module, embed_dim = build_module(cfg, ds_config)

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})", flush=True)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})", flush=True)
    module.eval().to(device)
    encoder = resolve_encoder(module)
    print(f"[{model}/{dataset}] built+loaded in {time.time()-t_build:.1f}s embed_dim={embed_dim}", flush=True)

    loader = _build_loader(split, model, dataset, cfg, ds_config)
    if loader is None:
        print(f"[{model}/{dataset}] no {split} split; skip", flush=True)
        return None

    t_extract = time.time()
    E, y = extract_embeddings(encoder, loader, device, MAX_SAMPLES)
    extract_secs = time.time() - t_extract

    rm = rankme(E)
    ld_val, n_cls_used, spc = lidar(E, y, samples_per_class=LIDAR_SAMPLES_PER_CLASS)
    results = {
        "rankme": rm,
        "lidar": ld_val,
        "lidar_n_classes_used": n_cls_used,
        "lidar_samples_per_class": spc,
        "n_samples": int(E.shape[0]),
        "n_classes": int(y.unique().numel()),
        "embed_dim": int(E.shape[1]),
        "extract_seconds": float(extract_secs),
    }

    print(
        f"[{model}/{dataset}/{split}] rankme={rm:.2f}  lidar={ld_val}"
        f"  n={E.shape[0]} classes_used={n_cls_used}/{results['n_classes']}"
        f"  ({extract_secs:.1f}s)",
        flush=True,
    )

    if not dry_run:
        cfg_record = {
            "backbone": BACKBONE,
            "max_samples": MAX_SAMPLES,
            "lidar_samples_per_class": LIDAR_SAMPLES_PER_CLASS,
        }
        if split == "train":
            cfg_record["loader"] = "train_with_val_transform"
        mgr.append(
            model=model,
            dataset=dataset,
            seed=None,
            split=split,
            checkpoint=str(ckpt),
            config=cfg_record,
            results=results,
            overwrite=overwrite,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="which split to extract embeddings from. val uses the model's "
                             "default transforms; train uses the val transform + single-view "
                             "collate so train images are processed identically to val.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--shard-dir", type=Path, default=None,
        help="write results to a per-(model,dataset) JSON shard here instead of the "
             "main history file. Required for parallel SLURM runs to avoid write races.",
    )
    args = parser.parse_args()

    if args.shard_dir is not None:
        args.shard_dir.mkdir(parents=True, exist_ok=True)
        if args.model and args.dataset:
            shard_path = args.shard_dir / f"{args.model}_{args.dataset}.json"
        else:
            # --all + --shard-dir would mean one giant shard; that defeats the purpose
            parser.error("--shard-dir requires --model + --dataset (one shard per task)")
        mgr = ExperimentManager(shard_path)
    else:
        mgr = ExperimentManager(HISTORY_PATHS[args.split])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}  split = {args.split}  history = {mgr.path}", flush=True)

    if args.all:
        targets = discover_all()
    elif args.model and args.dataset:
        targets = [(args.model, args.dataset)]
    else:
        parser.error("pass --model+--dataset or --all")

    print(f"targets ({len(targets)}): {targets[:5]}{'…' if len(targets)>5 else ''}", flush=True)
    for model, dataset in targets:
        try:
            process_one(model, dataset, args.split, device, mgr,
                        overwrite=args.overwrite, dry_run=args.dry_run)
        except Exception as e:
            print(f"[{model}/{dataset}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
