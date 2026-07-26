#!/bin/bash
#SBATCH --job-name=prewarm
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=./benchmark-runs/prewarm_%j.out
#SBATCH --error=./benchmark-runs/prewarm_%j.err

set -euo pipefail
cd .
exec ./.venv/bin/python -m benchmarks.prewarm "$@"
