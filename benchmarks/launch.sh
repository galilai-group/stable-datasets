#!/bin/bash
# Launch SSL benchmark sweep on SLURM.
#
# Usage:
#   ./launch.sh cifar10
#   ./launch.sh cifar10,stl10,flowers102
#   MODELS=simclr,dino ./launch.sh cifar10,imagenet
#   CONFIG=local_parallel ./launch.sh cifar10
#   SLURM_PARTITION=a100 SLURM_QOS=high ./launch.sh imagenet
#
# Environment variables:
#   CONFIG          - Hydra config name: slurm, local_parallel, config (default: slurm)
#   MODELS          - Comma-separated model list (default: all 7 incl. supervised)
#   BACKBONES       - Comma-separated timm model names (default: vit_small_patch16_224)
#   SLURM_PARTITION - SLURM partition (default: gpu)
#   SLURM_QOS       - SLURM QOS (default: normal)

set -euo pipefail

# Detect project root: benchmarks/launch.sh → repo root is one level up.
# Avoids relying on `git rev-parse` (fails when this is a worktree whose
# parent .git has moved).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PROJECT_ROOT

DATASETS=${1:?"Usage: launch.sh <dataset1,dataset2,...> [extra hydra overrides...] [env: CONFIG=... MODELS=... BACKBONES=...]"}
shift
CONFIG=${CONFIG:-local_parallel}
MODELS=${MODELS:-supervised,simclr,dino,mae,lejepa,nnclr,barlow_twins}
BACKBONES=${BACKBONES:-vit_small_patch16_224}

echo "=== Benchmark Sweep ==="
echo "Project:   $PROJECT_ROOT"
echo "Config:    $CONFIG"
echo "Datasets:  $DATASETS"
echo "Models:    $MODELS"
echo "Backbones: $BACKBONES"
echo "=========================="

"$PROJECT_ROOT/.venv/bin/python" -m benchmarks.run \
    --multirun \
    --config-name "$CONFIG" \
    "dataset=$DATASETS" \
    "model=$MODELS" \
    "backbone=$BACKBONES" \
    "$@"
