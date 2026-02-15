#!/bin/bash
# Launch SSL benchmark sweep on SLURM.
#
# Usage:
#   ./launch.sh cifar10
#   ./launch.sh cifar10,stl10,flowers102
#   MODELS=simclr,dino BACKBONES=resnet18,vit_tiny ./launch.sh cifar10,imagenet
#   CONFIG=local_parallel ./launch.sh cifar10
#   SLURM_PARTITION=a100 SLURM_QOS=high ./launch.sh imagenet
#
# Environment variables:
#   CONFIG          - Hydra config name: slurm, local_parallel, config (default: slurm)
#   MODELS          - Comma-separated model list (default: all 6)
#   BACKBONES       - Comma-separated backbone list (default: resnet18,vit_tiny)
#   SLURM_PARTITION - SLURM partition (default: gpu)
#   SLURM_QOS       - SLURM QOS (default: normal)

set -euo pipefail

# Detect project root (git repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../..")"
export PROJECT_ROOT

DATASETS=${1:?"Usage: launch.sh <dataset1,dataset2,...> [env: CONFIG=... MODELS=... BACKBONES=...]"}
CONFIG=${CONFIG:-slurm}
MODELS=${MODELS:-simclr,dino,mae,lejepa,nnclr,barlow_twins}
BACKBONES=${BACKBONES:-resnet18,vit_tiny}

echo "=== SSL Benchmark Sweep ==="
echo "Project:   $PROJECT_ROOT"
echo "Config:    $CONFIG"
echo "Datasets:  $DATASETS"
echo "Models:    $MODELS"
echo "Backbones: $BACKBONES"
echo "=========================="

python -m stable_datasets.benchmarks.self_supervised.main \
    --multirun \
    --config-name "$CONFIG" \
    "dataset=$DATASETS" \
    "model=$MODELS" \
    "backbone=$BACKBONES"
