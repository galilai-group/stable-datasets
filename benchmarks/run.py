"""Hydra entry point for benchmark runs.

Runs any combination of {model, backbone, dataset} with config-driven
hyperparameters.  See launch.sh for usage examples.
"""

from __future__ import annotations

import logging
import math
import os

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint

from benchmarks.gdrive_checkpoint import (
    GoogleDriveModelCheckpoint,
    resolve_gdrive_cfg,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict

import wandb
from benchmarks.dataset import create_dataset, get_config
from benchmarks.models import (
    build_module,
    create_eval_callbacks,
    get_transforms,
    resolve_backbone_family,
)


log = logging.getLogger(__name__)


# Param resolution


def _assert_supported_combo(cfg: DictConfig) -> None:
    """Reject method/backbone pairs the grid does not define.

    MAE reconstructs masked *patch tokens*, so it is meaningful only on a ViT
    encoder — there is no ResNet MAE in this benchmark. Without this guard a
    `MODELS=mae BACKBONES=resnet50` sweep launches, burns a GPU slot, and dies
    somewhere inside the decoder with an opaque shape error hours later (or
    worse, submitit swallows it and the job reports success with no metrics).
    """
    from benchmarks.models import resolve_backbone_family, resolve_backbone_name

    if cfg.model.name == "mae" and resolve_backbone_family(cfg.backbone) == "resnet":
        raise ValueError(
            f"mae + {resolve_backbone_name(cfg.backbone)!r} is not a supported combination: "
            "MAE needs patch tokens from a ViT encoder. The ResNet-50 half of the grid "
            "runs the other 6 methods (supervised, simclr, dino, lejepa, nnclr, barlow_twins)."
        )


def _resolve_params(cfg: DictConfig) -> None:
    """Merge per-dataset params from model config into training config.

    Priority: CLI overrides > dataset-specific > model defaults > config.yaml.
    Mutates cfg in place.
    """
    params = cfg.model.get("params", {})
    ds_params = params.get(cfg.dataset, {})
    default_params = params.get("default", {})

    cli_overrides = set()
    try:
        for override in HydraConfig.get().overrides.task:
            cli_overrides.add(override.split("=")[0])
    except ValueError:
        pass

    def _pick(key, cfg_key, fallback):
        if cfg_key in cli_overrides:
            return fallback
        return ds_params.get(key, default_params.get(key, fallback))

    with open_dict(cfg):
        cfg.training.batch_size = _pick("batch_size", "training.batch_size", cfg.training.batch_size)
        cfg.training.max_epochs = _pick("max_epochs", "training.max_epochs", cfg.training.max_epochs)
        cfg.training.accumulate_grad_batches = _pick(
            "accumulate_grad_batches",
            "training.accumulate_grad_batches",
            cfg.training.get("accumulate_grad_batches", 1),
        )
        accum = cfg.training.accumulate_grad_batches
        if accum > 1:
            cfg.training.batch_size = cfg.training.batch_size // accum

        # Central benchmark epoch table wins over the per-model params block
        # (but never over an explicit CLI override) so all 7 methods share one
        # citation-pinned budget per dataset. See conf/config.yaml.
        _be = cfg.get("benchmark_epochs", None)
        if _be is not None and _be.get("enabled", False) and "training.max_epochs" not in cli_overrides:
            _base = (_be.get("base", {}) or {}).get(cfg.dataset, None)
            if _base is not None:
                _mult = int(_be.get("mae_multiplier", 4)) if cfg.model.name == "mae" else 1
                cfg.training.max_epochs = int(_base) * _mult
            else:
                # Loud, because the fallback is silent and wrong-looking: a dataset
                # absent from the table drops through to the model's params block and
                # ultimately to config.yaml's training.max_epochs (50) — which would
                # quietly produce a 50-epoch "benchmark" run next to 400-epoch ones.
                log.warning(
                    "[epochs] %r is NOT in benchmark_epochs.base — falling back to "
                    "max_epochs=%d. Add it to conf/config.yaml if this is a benchmark "
                    "dataset; ignore if it is an ad-hoc run.",
                    cfg.dataset, cfg.training.max_epochs,
                )

        lr_override = ds_params.get("lr", default_params.get("lr", None))
        if lr_override is not None:
            cfg.model._lr_override = float(lr_override)

        # Per-dataset model-level overrides (DINO centering/teacher EMA + SK flag).
        # Raised momenta or SK help stabilize centering on small-batch/few-class
        # datasets.
        for key, caster in (
            ("center_momentum", float),
            ("momentum_teacher", float),
            ("sinkhorn_knopp", bool),
        ):
            if key in cli_overrides or f"model.{key}" in cli_overrides:
                continue
            override = ds_params.get(key, default_params.get(key, None))
            if override is not None:
                cfg.model[key] = caster(override)


# W&B logger


def _create_wandb_logger(cfg: DictConfig, seed: int | None, n_params: int | None = None) -> WandbLogger:
    run_name = f"{cfg.model.name}_{cfg.backbone}_{cfg.dataset}"
    if seed is not None:
        run_name += f"_seed{seed}"

    # Family tag ("vit"/"resnet") gives one-click backbone filtering in the W&B
    # UI, complementing the config.backbone filter that render_latex.py uses.
    family = resolve_backbone_family(cfg.backbone)
    tags = [family]
    if seed is not None:
        tags.append("seed")
    run_tag = cfg.get("run_tag", None)
    if run_tag is not None:
        tags.append(str(run_tag))

    # Backbone-appropriate optimizer block (mirrors build_optim_config).
    opt_key = f"{family}_optimizer"
    if hasattr(cfg.model, opt_key):
        lr = getattr(cfg.model, opt_key).lr
    elif hasattr(cfg.model, "vit_optimizer"):
        lr = cfg.model.vit_optimizer.lr
    else:
        lr = cfg.model.optimizer.lr

    # Keep W&B local run data on SCRATCH, not HOME: os.getcwd() is the hydra output
    # dir under the repo (HOME, 100 GB quota), and W&B caches quickly fill it. Derive
    # a scratch path from checkpoint.dir (already resolved to scratch).
    wandb_save_dir = os.path.join(os.path.dirname(str(cfg.checkpoint.dir)), "wandb-local")
    os.makedirs(wandb_save_dir, exist_ok=True)
    return WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=run_name,
        id=wandb.util.generate_id(),
        log_model=False,
        save_dir=wandb_save_dir,
        tags=tags or None,
        config={
            "model": cfg.model.name,
            "backbone": cfg.backbone,
            "backbone_family": family,
            "n_params": n_params,
            "dataset": cfg.dataset,
            "lr": lr,
            "batch_size": cfg.training.batch_size,
            "accumulate_grad_batches": cfg.training.accumulate_grad_batches,
            "max_epochs": cfg.training.max_epochs,
            "seed": seed,
        },
    )


# GPU assignment for local parallel runs


def _assign_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return
    try:
        job_num = HydraConfig.get().job.num
    except ValueError:
        return
    gpu_id = job_num % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log.info(f"Job #{job_num} pinned to GPU {gpu_id}/{num_gpus}")


# Main


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for SLURM/local runs. The Modal backend
    (benchmarks.modal_app) composes a cfg and calls train(cfg) directly, so the
    training core stays in one place across all execution backends."""
    train(cfg)


def train(cfg: DictConfig, extra_callbacks: list | None = None) -> None:
    # extra_callbacks lets a backend inject Lightning callbacks without run.py
    # knowing about it — e.g. the Modal backend passes a Volume-commit callback so
    # checkpoints persist across preemptions. The SLURM/local path passes none.
    # Disable spt's run-registry / cache_dir on the SLURM worker. Module-level
    # placement is unreliable here because hydra-submitit-launcher unpickles
    # `main` directly and the spt singleton is created during plugin discovery
    # before benchmarks.run is re-imported. Setting this inside main() runs
    # before `spt.Manager(...)` is constructed below, which is when
    # `_resolve_run_dir` reads cache_dir. With cache_dir=None, manager.py:469
    # short-circuits and Lightning's ModelCheckpoint at cfg.checkpoint.dir is
    # the only path that gets written.
    spt.get_config().cache_dir = None
    log.info(f"spt cache_dir disabled (was: {spt.get_config().cache_dir!r})")

    if cfg.get("distribute_gpus", False):
        _assign_gpu()

    seed = cfg.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    _assert_supported_combo(cfg)
    _resolve_params(cfg)

    log.info(
        f"{cfg.model.name} | {cfg.backbone} | {cfg.dataset} | "
        f"bs={cfg.training.batch_size} | accum={cfg.training.accumulate_grad_batches} | "
        f"epochs={cfg.training.max_epochs}"
    )

    # Data
    ds_config = get_config(cfg.dataset)
    if ds_config.modality != "image":
        raise ValueError(
            f"Dataset '{cfg.dataset}' is registered as modality={ds_config.modality!r}, "
            "but the current benchmark runner only has image model/backbone support."
        )
    train_transform, val_transform, collate_fn = get_transforms(cfg.model.name, ds_config, cfg.model)
    data, ds_config = create_dataset(
        cfg.dataset,
        train_transform,
        val_transform,
        collate_fn,
        cfg.training,
        data_dir=cfg.get("data_dir"),
    )

    # Scheduler needs total steps
    accum = cfg.training.accumulate_grad_batches
    total_steps = math.ceil(len(data.train) / accum) * cfg.training.max_epochs
    with open_dict(cfg):
        cfg.model._total_steps = total_steps

    # Model
    module, embed_dim = build_module(cfg, ds_config)

    # Backbone parameter count — for fair ViT-vs-ResNet capacity comparison.
    # Headless (num_classes=0): ViT-S/16 ~21.7M vs ResNet-50 ~23.5M. Logged to W&B below.
    n_params = sum(p.numel() for p in module.backbone.parameters())
    log.info(f"Backbone {cfg.backbone}: {n_params:,} parameters")

    # Callbacks
    # model_name routes the probe head through that model's own build_probe();
    # every model returns the same common protocol by default (DECISION_LOG #7).
    callbacks = create_eval_callbacks(
        module, ds_config, embed_dim,
        model_name=cfg.model.name,
        probe_protocol=cfg.get("probe_protocol", "common"),
    )
    ckpt_cfg = cfg.checkpoint
    run_dir_name = f"{cfg.model.name}_{cfg.backbone}_{cfg.dataset}"
    if seed is not None:
        run_dir_name += f"_seed{seed}"
    run_ckpt_dir = os.path.join(ckpt_cfg.dir, run_dir_name)
    # Run from the per-cell checkpoint dir so spt.Manager reads/writes its
    # wandb_resume.json sidecar THERE (unique per model/backbone/dataset/seed),
    # not in one shared file. os.getcwd() defaults to the repo root (hydra doesn't
    # chdir), so otherwise every concurrent run clobbers a single shared
    # wandb_resume.json and reads back another run's id — silently resuming into
    # the WRONG W&B run (cross-dataset contamination). A per-cell cwd keeps the
    # sidecar isolated yet stable across SLURM requeues of the same cell.
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.chdir(run_ckpt_dir)
    ckpt_kwargs = {
        "dirpath": run_ckpt_dir,
        "filename": "{epoch}-{step}",
        "every_n_epochs": ckpt_cfg.every_n_epochs,
        "save_last": ckpt_cfg.save_last,
        "monitor": ckpt_cfg.monitor,
        "mode": ckpt_cfg.mode,
        "save_top_k": ckpt_cfg.save_top_k,
        # Must be False so the checkpoint includes optimizer/scheduler/loop state
        # — Trainer.fit(ckpt_path=...) raises KeyError on resume otherwise, and
        # the error gets swallowed by submitit so the job appears to "complete"
        # without actually training. Discovered the hard way; see git log.
        "save_weights_only": False,
    }
    # EMA teacher update. TeacherStudentWrapper's teacher parameters ONLY move
    # when something calls update_teacher() -- the wrapper does not self-update on
    # forward. Without this callback the teacher stays frozen at its warm-init
    # copy of the student's RANDOM initial weights for the entire run, and DINO
    # silently degenerates into distilling a fixed random target. It trains, it
    # logs a falling loss, and the probe numbers look plausible, which is exactly
    # why this went unnoticed. Duck-typed on update_teacher so any future EMA
    # method picks it up automatically.
    #
    # Defaults are correct under gradient accumulation: update_after_backward is
    # False, so it fires on_train_batch_end guarded by trainer.global_step, which
    # advances once per OPTIMIZER step -- one EMA update per optimizer step, not
    # one per micro-batch (DINO/LeJEPA run accumulate_grad_batches=8).
    if any(hasattr(m, "update_teacher") and callable(m.update_teacher) for m in module.modules()):
        callbacks.append(spt.TeacherStudentCallback())
        log.info("[ema] registered TeacherStudentCallback (EMA teacher will be updated)")

    # Drive offload: when checkpoint.gdrive is configured for $SDS_WHOAMI we
    # REPLACE ModelCheckpoint rather than adding alongside it — two checkpoint
    # callbacks on the same dirpath would both write {epoch}-{step}.ckpt and
    # fight over top-k rotation, and the stock one would keep re-filling the
    # disk we are trying to drain. resolve_gdrive_cfg() returns None (and logs
    # why) whenever the config, $SDS_WHOAMI, or the rclone binary is missing,
    # so an unconfigured collaborator silently gets the normal local behaviour.
    _gd = resolve_gdrive_cfg(cfg)
    if _gd is not None:
        log.info(
            "[gdrive] checkpoints -> %s:%s (user=%s)",
            _gd["remote"], _gd.get("remote_dir", "checkpoints"), os.environ.get("SDS_WHOAMI"),
        )
        callbacks.append(GoogleDriveModelCheckpoint(**_gd, **ckpt_kwargs))
    else:
        callbacks.append(ModelCheckpoint(**ckpt_kwargs))
    # Backend-injected callbacks are appended AFTER ModelCheckpoint so their
    # on_validation_end fires after the checkpoint is written (e.g. the Modal
    # Volume-commit callback persists the freshly-saved ckpt).
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Auto-resume: if a previous SLURM walltime-out / requeue / manual resubmit
    # left a last.ckpt at the same (model, backbone, dataset, seed) path, hand
    # it to spt.Manager so trainer.fit() picks up where it stopped.
    # Safety check: weights-only checkpoints (legacy from save_weights_only=True)
    # cause Trainer.fit(ckpt_path=...) to raise KeyError on optimizer state, which
    # submitit silently swallows. Verify optimizer_states present before resuming.
    resume_ckpt = os.path.join(run_ckpt_dir, "last.ckpt")
    if os.path.isfile(resume_ckpt):
        try:
            _ckpt_peek = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            if "optimizer_states" not in _ckpt_peek:
                log.warning(
                    f"Skipping resume: {resume_ckpt} is weights-only (no optimizer state). "
                    f"Starting fresh."
                )
                resume_ckpt = None
            else:
                # Library-drift key remap: paper-era MAE checkpoints name the decoder's
                # MLP pre-norm `norm3` (older stable_pretraining reserved `norm2` for a
                # cross-attn slot); the current release renamed it `norm2` (timm-style).
                # Rename such keys IN PLACE so those checkpoints resume cleanly instead of
                # being discarded as "incompatible". Guarded on an exact name+shape match
                # against the current model, so it is a strict no-op for every other
                # checkpoint/method. This proven equivalent (same pre-norm feeding the MLP)
                # keeps the paper MAE backbones continuable rather than restarting fresh.
                _msd = module.state_dict()
                _csd = _ckpt_peek.get("state_dict", {})
                _remap = [
                    k for k in list(_csd)
                    if ".norm3." in k and k not in _msd
                    and k.replace(".norm3.", ".norm2.") in _msd
                    and k.replace(".norm3.", ".norm2.") not in _csd
                    and tuple(_msd[k.replace(".norm3.", ".norm2.")].shape) == tuple(_csd[k].shape)
                ]
                for _k in _remap:
                    _csd[_k.replace(".norm3.", ".norm2.")] = _csd.pop(_k)
                if _remap:
                    log.info(
                        f"Resume: remapped {len(_remap)} legacy decoder norm3->norm2 tensor(s) "
                        f"for checkpoint compatibility."
                    )
                # Shape-compat guard. Some online eval-callback queue buffers are
                # lazily shaped — e.g. `callbacks_modules.ordered_queue_label.out`
                # is [N] once a forward pass has run but [N, 1] in a fresh model —
                # and with num_sanity_val_steps=0 no forward happens before the
                # resume load, so Trainer.fit(ckpt_path=...) raises a shape-mismatch
                # RuntimeError mid-fit. submitit/Modal surface that as a hard job
                # failure. Verify every checkpoint tensor matches the current
                # model's shape; if any don't, start fresh instead of crashing.
                _model_sd = module.state_dict()
                _ckpt_sd = _ckpt_peek.get("state_dict", {})
                _bad = [
                    k for k, v in _ckpt_sd.items()
                    if k not in _model_sd or tuple(_model_sd[k].shape) != tuple(v.shape)
                ]
                if _bad:
                    # Distinguish disposable online-eval callback state (kNN/probe
                    # queue buffers + online linear probe, under callbacks_modules.*)
                    # from real model-weight mismatches. The eval buffers are lazily
                    # shaped, so they always "mismatch" a fresh module on resume — but
                    # they carry no training signal (they re-warm in a few epochs).
                    # Resetting them lets us resume model+optimizer+scheduler+epoch,
                    # instead of silently discarding all prior training.
                    _bad_model = [k for k in _bad if not k.startswith("callbacks_modules.")]
                    if _bad_model:
                        log.warning(
                            f"Skipping resume: {resume_ckpt} incompatible with current model "
                            f"({len(_bad_model)} model tensor(s) mismatch, e.g. {_bad_model[0]}). "
                            f"Starting fresh."
                        )
                        resume_ckpt = None
                    else:
                        _sd = _ckpt_peek["state_dict"]
                        # Strip ONLY the lazily-shaped online-eval QUEUE buffers (the kNN/probe
                        # memory bank + RankMe/LiDAR queues), which don't exist in a fresh module
                        # at checkpoint-restore time and so shape-mismatch. KEEP
                        # callbacks_modules.linear_probe.* — OnlineProbe.configure_model creates
                        # it eagerly (before Lightning restores the checkpoint), so it restores
                        # cleanly and the reported online linear-probe metric stays CONTINUOUS
                        # across the resume (no reset-and-rewarm dip).
                        _stripped = [k for k in _sd if k.startswith("callbacks_modules.") and "queue" in k.lower()]
                        for _k in _stripped:
                            del _sd[_k]
                        for _k in [
                            k for k in _ckpt_peek.get("callbacks", {})
                            if any(t in str(k) for t in ("Queue", "KNN", "RankMe", "LiDAR"))
                        ]:
                            del _ckpt_peek["callbacks"][_k]
                        # Name must end in ".ckpt": spt.Manager runs
                        # Path(ckpt_path).with_suffix(".ckpt"), so a "…last.ckpt.resume-clean"
                        # name would be rewritten to a nonexistent "…last.ckpt.ckpt".
                        resume_ckpt = os.path.join(os.path.dirname(resume_ckpt), "last.resume-clean.ckpt")
                        torch.save(_ckpt_peek, resume_ckpt)
                        # Tolerate the now-missing eval buffers (module keeps its fresh
                        # ones); model/optimizer/scheduler restore as normal.
                        module.strict_loading = False
                        # Continue the ORIGINAL W&B run rather than opening a new one:
                        # the run id is embedded in the checkpoint. spt.Manager reads
                        # wandb_resume.json from CWD (legacy mode) and injects the id
                        # before wandb.init, so the epoch axis stays continuous.
                        _wb = _ckpt_peek.get("wandb")
                        _sidecar = os.path.join(os.getcwd(), "wandb_resume.json")
                        if _wb and _wb.get("id"):
                            import json as _json
                            with open(_sidecar, "w") as _f:
                                _json.dump(_wb, _f)
                            log.info(f"Wrote wandb_resume.json (id={_wb['id']}) to continue the original run.")
                        elif os.path.isfile(_sidecar):
                            # Fresh-run resume (checkpoint carries no embedded wandb id): drop any
                            # stale sidecar left in this cell's dir so the Manager opens a NEW run
                            # instead of injecting a leftover id.
                            os.remove(_sidecar)
                            log.info("Removed stale wandb_resume.json — starting a fresh W&B run.")
                        log.info(
                            f"Resume: reset {len(_stripped)} eval-queue buffer(s), PRESERVED online "
                            f"linear probe, resumed model+optimizer+scheduler from epoch "
                            f"{_ckpt_peek.get('epoch')}."
                        )
                elif _remap:
                    # Only a legacy key rename was needed (no eval-buffer mismatch to
                    # strip): persist the remapped state_dict so the checkpoint Trainer
                    # actually loads carries norm2 keys, then resume from it.
                    resume_ckpt = os.path.join(
                        os.path.dirname(resume_ckpt), "last.resume-clean.ckpt"
                    )
                    torch.save(_ckpt_peek, resume_ckpt)
                    log.info(
                        f"Resume: saved norm3->norm2 remapped checkpoint; resumed "
                        f"model+optimizer+scheduler from epoch {_ckpt_peek.get('epoch')}."
                    )
            del _ckpt_peek
        except Exception as e:
            log.warning(f"Skipping resume: failed to load {resume_ckpt}: {e}")
            resume_ckpt = None
    else:
        resume_ckpt = None

    # Logger
    smoke_test = cfg.get("smoke_test", False)
    logger = True
    if cfg.wandb.enabled and not smoke_test:
        logger = _create_wandb_logger(cfg, seed, n_params=n_params)

    # Trainer
    has_val = data.val is not None
    # Under @hydra.main the runtime output dir is set; under the Modal backend
    # (hydra.compose, no HydraConfig singleton) it isn't — fall back to cwd.
    try:
        output_dir = HydraConfig.get().runtime.output_dir or os.getcwd()
    except ValueError:
        output_dir = os.getcwd()

    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else cfg.training.max_epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=accum,
        callbacks=callbacks,
        logger=logger,
        num_sanity_val_steps=0,
        limit_train_batches=3 if smoke_test else 1.0,
        limit_val_batches=3 if smoke_test else (1.0 if has_val else 0),
        accelerator="auto",
        default_root_dir=output_dir,
    )

    try:
        if resume_ckpt is not None:
            log.info(f"Resuming from checkpoint: {resume_ckpt}")
        manager = spt.Manager(trainer=trainer, module=module, data=data, ckpt_path=resume_ckpt)
        manager()
    finally:
        if cfg.wandb.enabled and not smoke_test:
            wandb.finish()


# dataset=all expansion (must run before Hydra parses argv)


def _expand_dataset_all():
    import sys

    from benchmarks.dataset import get_image_dataset_names

    for i, arg in enumerate(sys.argv):
        if arg.startswith("dataset=") and arg.split("=", 1)[1].lower() == "all":
            sys.argv[i] = f"dataset={','.join(get_image_dataset_names(include_results_only=True))}"
            if "--multirun" not in sys.argv and "-m" not in sys.argv:
                sys.argv.append("--multirun")
            break


_expand_dataset_all()

if __name__ == "__main__":
    main()
