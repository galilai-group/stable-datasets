#!/bin/bash
#SBATCH --job-name=prewarm
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
# Logs go to SCRATCH, not the repo: HOME is a 100 GB quota that this project has
# already filled, and a silently failed write there is a classic cause of jobs
# dying for no visible reason.
#SBATCH --output=/oscar/scratch/sboughan/stable-datasets-iclr/benchmark-runs/prewarm_%j.out
#SBATCH --error=/oscar/scratch/sboughan/stable-datasets-iclr/benchmark-runs/prewarm_%j.err

# Dataset building is CPU/IO-heavy and takes hours. It must NOT run on a login
# or vscode node: those are shared and cgroup-throttled, which drags the per-image
# processing rate down by orders of magnitude (cub200 took 1h53m for 5,994 images
# there) and is liable to get killed by CCV.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
exec ./.venv/bin/python -m benchmarks.prewarm "$@"
