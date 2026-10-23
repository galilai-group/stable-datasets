#!/bin/bash
#SBATCH --job-name=object-lejepa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=object-lejepa-%j.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch; do not train on a login node.}"
: "${OBJECT_DATASETS_PYTHON:?Set this to the absolute path of the environment Python executable.}"
: "${OBJECT_DATASETS_CACHE:?Set this to a writable cache root on scratch storage.}"
: "${OBJECT_DATASETS_RESULTS:?Set this to a writable experiment output root.}"
export MPLCONFIGDIR="${OBJECT_DATASETS_RESULTS}/matplotlib"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SLURM_SUBMIT_DIR}"
for seed in 42 43 44; do
    for dataset in eth80 freiburg; do
        srun "${OBJECT_DATASETS_PYTHON}" examples/object_datasets_lejepa.py \
            --dataset "${dataset}" --seed "${seed}" --device cuda \
            --cache-root "${OBJECT_DATASETS_CACHE}" \
            --output "${OBJECT_DATASETS_RESULTS}/${dataset}-seed${seed}-${SLURM_JOB_ID}"
    done
done
