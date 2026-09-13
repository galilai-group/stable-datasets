# ICLR grid — runbook

How to actually run the benchmark. For *why* it is set up this way, see
[DECISION_LOG.md](DECISION_LOG.md).

## 0. One-time shell setup

```bash
# Identifies you to the per-collaborator config blocks. Without it you still get
# normal local checkpointing, just no per-user Drive settings.
echo 'export SDS_WHOAMI=<sami|ian|leyang>' >> ~/.bashrc

# Only if you have NOT been added to the samibg W&B entity (see "Gotchas").
echo 'export SDS_WANDB_ENTITY=<your-wandb-entity>' >> ~/.bashrc
```

Everything else derives from `$HOME`. Nothing needs editing per machine.

| env var | default | what it moves |
| --- | --- | --- |
| `STABLE_DATASETS_ROOT` | `~/scratch/stable-datasets-iclr` | datasets, checkpoints, run dirs |
| `SDS_WANDB_PROJECT` | `stable-datasets-iclr` | W&B project |
| `SDS_WANDB_ENTITY` | `samibg` | W&B entity |
| `SDS_WHOAMI` | *(unset)* | which collaborator block applies |
| `SLURM_PARTITION` / `SLURM_QOS` | `gpu` / `normal` | where jobs land — **override these** |

## 1. Prewarm the datasets (do this first, once)

Training jobs that find a cold cache build it themselves while holding a GPU,
and several source URLs are flaky. Build the cache once, on a compute node:

```bash
cd /path/to/stable-datasets-pyarrow
DS=imagenet100,imagenette,cifar10,cifar100,stl10,svhn,food101,country211,cub200,\
fgvcaircraft,flowers102,dtd,galaxy10,pathmnist,octmnist,tissuemnist,bloodmnist,\
dermamnist,organamnist,pneumoniamnist

sbatch benchmarks/prewarm.sbatch "$DS"
```

It is idempotent — a cached dataset returns in under a second, so re-running it
only fetches what is missing. Logs land in
`$STABLE_DATASETS_ROOT/benchmark-runs/prewarm_<jobid>.{out,err}`.

**Do not run this on a login or vscode node.** Those are shared and
cgroup-throttled; cub200 took 1h53m for 5,994 images there versus seconds per
thousand on a compute node, and CCV will kill long CPU jobs on login hosts.

Budget: `country211` alone is an 11 GB download.

## 2. Launch training

```bash
# ViT-S/16 — all 7 methods
SLURM_PARTITION=3090-gcondo SLURM_QOS=cs-3090-gcondo CONFIG=slurm \
  MODELS=supervised,simclr,dino,mae,lejepa,nnclr,barlow_twins \
  BACKBONES=vit_small_patch16_224 \
  ./benchmarks/launch.sh "$DS"

# ResNet-50 — 6 methods, NO mae (it needs ViT patch tokens; the run would be
# rejected at startup anyway)
SLURM_PARTITION=3090-gcondo SLURM_QOS=cs-3090-gcondo CONFIG=slurm \
  MODELS=supervised,simclr,dino,lejepa,nnclr,barlow_twins \
  BACKBONES=resnet50 \
  ./benchmarks/launch.sh "$DS"
```

Add `smoke_test=true` for a 3-train/3-val-batch sanity run (skips W&B entirely).

Epochs are **not** specified per launch: `benchmark_epochs` in
`conf/config.yaml` pins a citation-backed budget per dataset, MAE at 4x. A
dataset missing from that table logs a warning and falls back to 50 — if you see
that warning on a benchmark dataset, add it to the table rather than overriding
on the CLI.

## 3. Reading the results

`eval/linear_probe_top1_epoch` is the headline number. It is the **best of 9
probe heads** (3 LR scales x 3 weight decays), not a single probe — see
DECISION_LOG #14. Also logged:

- `eval/linear_probe_top1_mean_epoch` — mean across heads
- `eval/linear_probe_lr<x>_wd<y>_top1_epoch` — each head

If the winning head is at the edge of the grid (`lr10` or `lr0p1`), widen
`probe_sweep.lr_scales` — the sweep is not bracketing the optimum.

## Gotchas

**Use `3090-gcondo`.** The default `gpu`/`norm-gpu` QOS caps you at 2 concurrent
jobs, which serialises a 20-job array into uselessness.

**Pin the GPU feature for interactive work.** The `3090-gcondo` *partition* also
contains RTX A5000/A5500 nodes at ~22.0 GiB against the 3090's 24 GiB. DINO and
LeJEPA (8-view multicrop) sit right at that boundary. `conf/slurm.yaml` already
sets `constraint: geforce3090|l40s`; for an interactive session add it yourself:

```bash
interact -n 6 -m 128g -g 1 -q 3090-gcondo -f geforce3090 -t 12:00:00
```

**W&B entity.** `samibg` is a personal entity and personal entities cannot take
collaborators. Until everyone is on a shared W&B *team*, non-owners must set
`SDS_WANDB_ENTITY` to their own — which means results land in separate projects
and have to be merged by hand at analysis time. Worth resolving before the grid
finishes rather than after.

**Cancelling jobs.** A plain `scancel` on these submitit jobs makes them requeue.
Disable requeue first:

```bash
scontrol update jobid=<id> requeue=0
scancel --signal=KILL <id> && scancel <id>
```

**HOME fills up.** It is a 100 GB quota and this project has already hit it.
Everything the benchmark writes goes to scratch by default; if you see jobs
dying for no visible reason, run `checkquota` before anything else.
