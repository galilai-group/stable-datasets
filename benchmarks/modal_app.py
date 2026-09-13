"""Modal execution backend for the ResNet-50 SSL benchmark (small/cheap datasets).

Runs the SAME training core as SLURM/local (`benchmarks.run.train`) inside Modal
containers on >=24 GB GPUs, fanning out (method x dataset) across the small
datasets while the big/expensive datasets stay on SLURM.

W&B logging is a HARD REQUIREMENT: each container aborts if WANDB_API_KEY is
absent, so no run can silently execute unlogged.

One-time setup + run (see also leyang_claude_code.md / the chat runbook):
  1) modal secret create wandb WANDB_API_KEY=<your key>     # YOU create this (your credential)
  2) modal run benchmarks/modal_app.py::upload_data          # push ~14 GB cache -> Volume
  3) modal run benchmarks/modal_app.py::smoke                # 1 run, verify cache hit + W&B
  4) modal run benchmarks/modal_app.py                       # fan out all 60 runs
  5) modal volume get stable-datasets-small /.pretrain_checkpoints <oscar_dst>   # pull ckpts back
"""

from __future__ import annotations

import glob
import os
import pathlib

import modal
from omegaconf import OmegaConf

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent


def _load_modal_cfg():
    # Locally, __file__ is the real benchmarks/ dir. In the Modal container the
    # entry script is auto-mounted flat at /root/, so __file__-relative paths
    # miss — but add_local_dir places the package (incl. conf/) at /root/benchmarks.
    for p in (HERE / "conf" / "modal.yaml", pathlib.Path("/root/benchmarks/conf/modal.yaml")):
        if p.exists():
            return OmegaConf.load(p)
    raise FileNotFoundError(f"modal.yaml not found (looked under {HERE} and /root/benchmarks/conf)")


CFG = _load_modal_cfg()

# Oscar-side source of the prewarmed processed cache (for upload_data).
OSCAR_PROCESSED = os.path.join(
    os.environ.get("STABLE_DATASETS_ROOT", os.path.expanduser("~/scratch/stable-datasets-iclr")),
    "processed",
)

# Processed-cache dir prefix per dataset. Most are "{dataset}_default", but some
# deviate: MedMNIST variants share one builder ("medmnist_<name>"), and the base
# FGVC-Aircraft is the "variant" config. Training finds these by hash internally;
# upload_data just needs to ship the right dirs to the Volume.
CACHE_PREFIX = {
    "bloodmnist": "medmnist_bloodmnist",
    "dermamnist": "medmnist_dermamnist",
    "breastmnist": "medmnist_breastmnist",
    "fgvcaircraft": "fgvcaircraft_variant",
    "fgvcaircraft_family": "fgvcaircraft_family",
}


def _processed_dirs(dataset: str, src: str) -> list[str]:
    """Absolute paths of the train/test/validation processed dirs for a dataset."""
    prefix = CACHE_PREFIX.get(dataset, f"{dataset}_default")
    dirs: list[str] = []
    for split in ("train", "test", "validation"):
        dirs += glob.glob(f"{src}/{prefix}_{split}_*")
    return dirs

app = modal.App("resnet-ssl-benchmark")

# Pinned to the exact stack in the working uv.lock so Modal reproduces Oscar.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "timm==1.0.26",
        "lightning==2.6.1",
        "pyarrow==20.0.0",
        "pylance==4.0.1",
        "wandb==0.25.1",
        "hydra-core==1.3.2",
        "omegaconf",
        # stable_datasets builder deps — imported at package load (it imports every
        # dataset module up front, e.g. linnaeus5 -> rarfile). Matches pyproject.
        "rarfile",
        "gdown",
        "loguru",
        "h5py",
        "scipy",
        "scikit-learn",
        "pandas",
        "datasets",
        "filelock",
        # git dep from the lock (0.1.7.dev43+g6dfdc346f). If this pin fails to
        # resolve at build time, drop the "@6dfdc346f" suffix to take the tip.
        "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@6dfdc346f",
    )
    .add_local_dir(
        str(HERE),
        "/root/benchmarks",
        ignore=["figures/*", "results/*", "dataset_samples/*", "transfer/checkpoints/*",
                "**/__pycache__", "*.pyc"],
    )
    .add_local_dir(
        str(REPO / "stable_datasets"),
        "/root/stable_datasets",
        ignore=["**/__pycache__", "*.pyc"],
    )
)

vol = modal.Volume.from_name(CFG.volume, create_if_missing=True)
wandb_secret = modal.Secret.from_name(CFG.wandb_secret)


@app.function(
    gpu=CFG.gpu,
    image=image,
    volumes={CFG.cache_mount: vol},
    secrets=[wandb_secret],
    timeout=int(CFG.timeout_hours) * 3600,
    cpu=float(CFG.cpu),
    memory=int(CFG.memory_gb) * 1024,
    max_containers=int(CFG.max_concurrent),
)
def train_one(model: str, dataset: str) -> str:
    """Train one (model, dataset) ResNet-50 run and persist its checkpoint."""
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root/benchmarks")

    # HARD REQUIREMENT: refuse to run unlogged — these experiments must land in W&B.
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY missing in container — refusing to run unlogged. "
            "Create it with: modal secret create wandb WANDB_API_KEY=<key>"
        )

    # Point data reads AND checkpoint writes at the mounted Volume (mirrors the
    # Oscar layout: processed cache at $ROOT/processed, ckpts at $ROOT/.pretrain_checkpoints).
    os.environ["STABLE_DATASETS_ROOT"] = CFG.cache_mount
    os.environ["STABLE_DATASETS_CACHE_DIR"] = CFG.cache_mount
    os.environ["PROJECT_ROOT"] = "/root/benchmarks"

    from hydra import compose, initialize_config_dir
    from lightning.pytorch.callbacks import Callback

    from benchmarks.run import train

    class _VolumeCommit(Callback):
        """Persist checkpoints to the Modal Volume as they're written.

        Modal GPU functions are preemptible: on preemption Modal restarts the
        Function on the same input, and run.py's auto-resume picks up last.ckpt —
        but only if it was committed to the Volume. Committing right after each
        ModelCheckpoint save (this callback is appended AFTER it, so its
        on_validation_end runs later) caps the work lost to a preemption at
        every_n_epochs, instead of the whole run. The commit is fast because it
        flushes only what changed since the last commit.
        """

        def __init__(self, volume, every_n_epochs):
            self._vol = volume
            self._every = max(1, int(every_n_epochs))

        def on_validation_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % self._every == 0:
                self._vol.commit()

    with initialize_config_dir(config_dir="/root/benchmarks/conf", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model}",
                f"dataset={dataset}",
                f"backbone={CFG.backbone}",
                "wandb.enabled=true",
            ],
        )
    train(cfg, extra_callbacks=[_VolumeCommit(vol, cfg.checkpoint.every_n_epochs)])
    vol.commit()  # final flush of the checkpoint under /cache/.pretrain_checkpoints
    return f"ok {model}/{dataset}"


@app.local_entrypoint()
def main():
    """Fan out all (method x dataset) small-dataset runs on Modal."""
    jobs = [(m, d) for d in CFG.datasets for m in CFG.methods]
    print(f"Fanning out {len(jobs)} runs on Modal {CFG.gpu} (<= {CFG.max_concurrent} concurrent)")
    ok, fail = 0, 0
    for res in train_one.starmap(jobs, return_exceptions=True):
        if isinstance(res, Exception):
            fail += 1
            print(f"  FAILED: {res!r}")
        else:
            ok += 1
            print(f"  {res}")
    print(f"done: {ok} ok, {fail} failed  (checkpoints on Volume '{CFG.volume}':/.pretrain_checkpoints)")


@app.local_entrypoint()
def smoke():
    """One quick run to verify the image, cache-hit, and W&B logging before the fan-out."""
    m, d = CFG.methods[0], CFG.datasets[-1]  # simclr x beans (smallest)
    print(f"smoke: {m}/{d} on {CFG.gpu} — verifying cache hit + W&B logging")
    print(train_one.remote(m, d))


@app.local_entrypoint()
def upload_data(src: str = OSCAR_PROCESSED):
    """Push the 10 small datasets' processed cache from Oscar -> Modal Volume (/processed)."""
    total = 0
    with vol.batch_upload(force=True) as batch:
        for d in CFG.datasets:
            hits = _processed_dirs(d, src)
            if not hits:
                print(f"  WARNING: no processed cache found for {d} under {src} "
                      f"(prefix {CACHE_PREFIX.get(d, d + '_default')!r})")
                continue
            for p in hits:
                name = os.path.basename(p)
                print(f"  + {name}")
                batch.put_directory(p, f"/processed/{name}")
                total += 1
    print(f"uploaded {total} processed dirs to Volume '{CFG.volume}':/processed")
