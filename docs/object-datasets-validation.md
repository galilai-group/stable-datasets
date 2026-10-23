# ETH-80 and Freiburg Groceries: reproducible validation

`ETH80` and `FreiburgGroceries` provide image classification datasets with reproducible partitions, cached image decoding, and PyTorch integration. The benchmark evaluates LeJEPA representations with a frozen linear classifier and a matched random-encoder baseline. Dataset documentation contains five attributed preview images per dataset; full source archives and model checkpoints are not part of the library distribution.

## Implementation and dependencies

The builders use `BaseDatasetBuilder`, `DatasetSource`, `DownloadInfo`, `ClassLabel`, `Image`, the shared download/cache machinery, and `StableDataset.with_transform` / `with_format`. The dataset API baseline is stable-datasets commit `f38ac71a1ef3c97be034dbbf51861e66909122b3`.

The benchmark imports [LeJEPA from stable-pretraining at `ab836bf699a2be712dfcf980a5eb70d36391e876`](https://github.com/galilai-group/stable-pretraining/blob/ab836bf699a2be712dfcf980a5eb70d36391e876/stable_pretraining/methods/lejepa.py). Training uses `global_views` and `local_views`. The `vit_tiny_patch16_224` backbone supplies the `embed_dim` attribute required by this implementation and produces 192-dimensional encoder features.

The loss combines invariance to the mean of the two global projections with `0.02 * SIGReg`. SIGReg operates on the concatenated projected views. The method reference is [Balestriero and LeCun, LeJEPA](https://arxiv.org/abs/2511.08544).

## Install

The recorded benchmark environment uses Python 3.12. The dataset library supports Python >=3.10. Installation from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r examples/requirements-object-datasets.txt
python -m pip install pytest ruff==0.15.0
```

Each run records package versions in `requirements-lock.txt` and `environment.json`. The pinned requirements file defines the benchmark dependencies; generated lock files also contain machine-specific editable-install paths. CUDA execution requires a PyTorch build compatible with the device driver. CPU and CUDA execution share the source and JSON configuration; floating-point results can vary across devices and software versions.

## Data protocol

| Dataset | SSL/probe training | Validation | Final test |
| --- | ---: | ---: | ---: |
| ETH-80 | 1,968 images / 48 objects | 656 images / 16 objects | 656 images / 16 objects |
| Freiburg, author split0 | 3,143 unique images | 786 unique images | 1,018 official test images |

ETH-80 uses a fixed custom 6/2/2 object split. Original object IDs are numerically ordered within category; cars have non-contiguous IDs. All views of an object stay together. This differs from the original leave-one-object-out protocol, so scores from those partitions are not directly comparable.

Freiburg's loader preserves all 7,425 official training rows, including repeated paths. For this experiment, the official split0 training list is deduplicated by path, sorted, then stratified with `test_size=0.2, random_state=42`. The official test set is unchanged. The validation split and deduplication are experimental choices, not author-provided metadata. Physical product identities are unavailable; only path-level separation can be verified.

SSL receives only augmented training images, without labels. No validation or test image is used to optimize the encoder. `splits.json` records the exact image IDs, labels, and ETH-80 object IDs for every partition. Loader metadata/integrity checks may inspect test labels and pixels; those checks do not optimize models or select parameters.

## Offline and real-data checks

```bash
pytest -q \
  stable_datasets/tests/images/test_eth80.py \
  stable_datasets/tests/images/test_freiburg_groceries.py \
  stable_datasets/tests/test_object_dataset_example.py

python examples/validate_object_datasets.py \
  --cache-root ./data/object-datasets \
  --output ./runs/object-datasets-integration.json
```

The dataset and benchmark test files contain 20 offline tests. Together with the related builder/cache/dataset/schema regressions, all 159 tests passed on macOS with Python 3.10.22 and Python 3.12.14. Compatibility testing uses separate dependency environments; the measured training results use Python 3.12.14.

The offline tests use synthetic archives, forbid network access during cache reopening, and cover non-contiguous ETH-80 IDs, all five Freiburg configurations, author label order, repeated training paths, malformed metadata, missing images, split isolation, and PyTorch batching. The experiment tests cover deduplication before validation splitting, label-free SSL inputs, and training-only feature normalization.

The integration script verifies SHA-256 values, decodes all source images, compares **every cached sample's RGB pixels** against its source, independently checks all five Freiburg manifests, confirms the split0 row order and duplicates, verifies split disjointness and ETH-80 object separation, reopens caches, and forms batches. Raw assets and default processed caches occupy approximately 2 GB. Downloads are cached for subsequent runs.

## Fixed training and evaluation protocol

The configuration is `examples/configs/object_datasets_lejepa.json`:

- Encoder: timm ViT-Tiny/16, 192-dimensional features, random initialization, no pretrained weights, no stochastic depth.
- Projector: 192 -> 256 -> 128, with BatchNorm and ReLU between layers.
- Views: two 64x64 global crops (scale 0.3--1.0) and two 32x32 local crops (scale 0.05--0.3), independently sampled per image. Each uses horizontal flipping, color jitter, and random grayscale. Input normalization is fixed mean/std 0.5 per channel, not fitted to test data.
- SIGReg: 64 slices, 17 integration points, `t_max=3`, lambda 0.02, using upstream code unchanged.
- Optimizer: AdamW, learning rate 0.0005, weight decay 0.05; two warmup epochs followed by cosine decay to 1% of the peak learning rate.
- Budget: 20 complete training epochs with `drop_last=True`, batch size 32; seeds 42, 43, 44. ETH-80 has 61 steps/epoch; Freiburg has 98.
- Frozen evaluation: resize to 64x64, extract encoder features in eval mode, and fit per-feature mean/std using training features only. All encoder parameters are frozen; the script asserts that every encoder state tensor is unchanged across the evaluation phase.
- Linear head: randomly initialized 192 -> number-of-classes linear layer, full-batch cross-entropy, AdamW at 0.01 for 200 iterations. Search weight decay in `[0, 0.001, 0.01]`; choose the iteration/decay with highest validation top-1 accuracy, taking the first in deterministic iteration order on ties. Do not refit on validation data.
- Baseline: an exact copy of the encoder before SSL training, including its initial weights. It gets the same data partitions, transformations, feature scaling rule, head initialization seed, optimizer, and validation search budget as the trained encoder.
- Final metrics: test top-1 accuracy, macro recall (balanced accuracy), correct/count, full confusion matrix, and per-image predictions. Test evaluation occurs after the head is selected. No training schedule or hyperparameter is changed based on a test result.

The benchmark fixes image resolution, epoch count, projector size and slice count across both datasets. The three seeds measure initialization and training variability on one fixed partition; they do not characterize variability across alternative dataset partitions. The input sizes and training budget differ from the original paper's benchmark configurations.

## Commands

Training smoke tests, with test evaluation disabled:

```bash
python examples/object_datasets_lejepa.py --dataset eth80 \
  --cache-root ./data/object-datasets --output ./runs/eth80-smoke \
  --epochs 1 --max-batches 5 --skip-probe
python examples/object_datasets_lejepa.py --dataset freiburg \
  --cache-root ./data/object-datasets --output ./runs/freiburg-smoke \
  --epochs 1 --max-batches 5 --skip-probe
```

Three-seed benchmark:

```bash
for seed in 42 43 44; do
  for dataset in eth80 freiburg; do
    python examples/object_datasets_lejepa.py \
      --dataset "$dataset" --seed "$seed" --device cpu \
      --cache-root ./data/object-datasets \
      --output "./runs/${dataset}-seed${seed}"
  done
done
```

Output directories must not already exist. Each run contains the resolved configuration, environment and package versions, exact split manifest, epoch losses and timings, final model/optimizer checkpoint, original random encoder checkpoint, selected heads and their normalization parameters, validation search traces, metrics, and test predictions. Checkpoints support model inspection and feature evaluation. Training resumption is not implemented.

## Slurm execution

`examples/object_datasets_oscar.sbatch` runs the same benchmark with `--device cuda` inside a Slurm allocation. It requests one GPU, four CPUs, 16 GB memory, and two hours on the `gpu` partition. Required environment variables identify the Python executable, dataset cache and results directory:

```bash
export OBJECT_DATASETS_PYTHON=/absolute/path/to/venv/bin/python
export OBJECT_DATASETS_CACHE=/absolute/path/to/scratch/object-datasets
export OBJECT_DATASETS_RESULTS=/absolute/path/to/scratch/object-results
mkdir -p "$OBJECT_DATASETS_RESULTS"
sbatch examples/object_datasets_oscar.sbatch
squeue -u "$USER"
```

Partition and account settings depend on the cluster allocation. The script requires `SLURM_JOB_ID` and validates its required environment variables. Validation covers ShellCheck and Bash syntax; CUDA execution and Oscar submission are not covered by the recorded experiments. Platform configuration is documented in the official [Oscar quickstart](https://docs.ccv.brown.edu/oscar/quickstart) and [GPU submission reference](https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu).

## Results

[Measured results](object-datasets-results.md) include aggregate and per-seed validation/test metrics, selected probes, timings, and evaluation limits. Full checkpoints and source archives are not included in the repository.
