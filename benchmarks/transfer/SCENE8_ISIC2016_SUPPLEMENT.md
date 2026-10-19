# Supplementary Verification Workflow

Run from the repository root. All commands use local result files and CPU FP32.
See `benchmarks/SCENE8_ISIC2016_EXPERIMENT_PROTOCOL.md` for the pre-model design,
limitations, and exact scientific interpretation.

## Environment

The executed environment uses Python 3.12.14, PyTorch 2.9.1+cpu,
torchvision 0.24.1+cpu, NumPy 2.5.2, Pillow 12.3.0, PyArrow 26.0.0,
scikit-learn 1.9.1, and SciPy 1.18.1. Install CPU PyTorch explicitly before
the repository dependencies, using a dedicated virtual environment:

```console
python -m venv .venv
python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . pytest ruff matplotlib
```

Activate the environment or replace `python` with its interpreter path. On
Windows this is `.venv\Scripts\python.exe`. Exact executed package versions are
recorded in the experiment evidence and the supplementary environment file.
The data audit and frozen transfer runner do not import stable-pretraining,
Lightning, timm, or W&B. The separate training smoke check uses the pinned
stable-pretraining revision documented below.

## Original assets and data audit

Download the original Scene8 ZIP as `.cache/project1/raw/scene8.zip`, and place
the four original ISIC2016 ZIP/CSV files next to it. Their canonical names and
source URLs are in the dataset documentation and `SOURCE` definitions. Then run:

```console
python -m benchmarks.audit_scene8_isic2016 --raw-dir .cache/project1/raw --cache-dir .cache/project1/processed --output-dir .cache/project1/verified-v2
```

This verifies asset hashes, source JPEG bytes, every decoded image, warm caches,
and manifests for seeds 42/43/44. Keep older manifests under their original
version names. Do not substitute the library's implicit test-as-validation
fallback for these manifests.

## Resource pilot and complete feature extraction

```console
python -m benchmarks.transfer.scene8_isic2016 pilot --threads 4 --batch-size 4
python -m benchmarks.transfer.scene8_isic2016 extract --threads 4 --batch-size 4
```

The first command processes 32 fitting images per dataset and saves
`pilot.json` plus source/crop montages in `.cache/project1/experiments/`.
Inspect both montages before model evaluation. Timing includes indexed image
decoding, source-pixel verification, preprocessing, and inference; initial model
loading and montage creation are outside that timer. Peak process memory includes
the pilot's model, native tensors, and loaded image resources. The full-run estimate
is a resource estimate, not a promised duration.

The pinned original DINO implementation and checkpoint are downloaded to the
ignored `.cache/project1/assets/dino/` directory when absent. Every file must
match `dino_vits16_assets.json`, which records source commit
`7c446df5b9f45747937fb0d72314eb9f7b66930a`. The loader uses the original implementation,
strictly loads all backbone weights, freezes parameters, and checks that forward
matches the final normalized CLS token. This avoids an unverified model-port
comparison. Source licenses remain alongside downloaded source files.

The extractor includes the final partial batch and compares each live sample
with its audited ID, label, dimensions, and decoded RGB fingerprint. A completed
cache is accepted only after checksum, metadata, ID ordering, labels, dimensions,
and finite/nonzero feature checks. It contains no resized source images.

Complete-source feature caches can be reused across manifest seeds because
inference has no fitted target-data statistics. Changing encoder assets,
preprocessing, source records, executed extraction code, relevant versions, or
thread configuration creates a new cache identity. Use the same `--threads`
setting for extraction and evaluation. Defaults are documented rather than
silently inferred from a directory name.

## Declared classifier matrix

```console
python -m benchmarks.transfer.scene8_isic2016 evaluate --threads 4
```

This runs 36 primary learned conditions, six full-budget official ISIC companions,
and 18 prior baselines. Each primary condition independently selects its declared
hyperparameter using only its allowed fitting/validation labels, then refits on
their selected union. Low-budget runs cannot refit on the parent full-label pool.
Companions reuse the filtered full-budget hyperparameter. Scores do not control
which conditions are retained.

Each condition stores its selected IDs, exact label counts including validation,
all validation candidate scores, classifier parameter, manifest and feature-cache
identity, test predictions, metrics, and elapsed time. The final `matrix.json`
is written only after all 60 conditions finish. Do not treat partial run folders
as a completed matrix.

## Recomputed summaries and report

```console
python -m benchmarks.transfer.summarize_scene8_isic2016
```

The summarizer verifies every prediction file and independently recomputes its
metrics before writing English artifacts under
`docs/source/datasets/experiments/scene8_isic2016/`:

- `REPORT.md`, mean/sample-SD tables, label-efficiency plots, and confusion matrices.
- `runs.csv` with every condition and its actual label budget.
- `predictions.csv.gz` with every source ID, target, class count, and probability.
- `matrix.json.gz` with all selected subsets and validation candidate scores.
- `errors_seed42.csv` with every full-budget primary error for the declared seed.
- `uncertainty.json` with declared seed-42 conditional exact-group bootstrap intervals.
- `evidence.json` binding the artifacts to source, manifest, cache, and environment hashes.

No raw photographs, Arrow caches, or model weights belong in a PR. Probability
columns follow the fixed dataset class order; unused columns in the combined
binary/multiclass prediction CSV are empty. Preserve full numeric precision when
recomputing metrics. Bootstrap intervals and seed SD quantify different sources
of variation; neither establishes patient or pretraining independence.

## Known-method optimization smoke check

The training dependency is pinned to
`ab836bf699a2be712dfcf980a5eb70d36391e876`, rather than floating main:

```console
python -m pip install "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining@ab836bf699a2be712dfcf980a5eb70d36391e876"
```

Keep the validated inference packages fixed while resolving its optional training
stack. Run the separate optimization check:

```console
python -X utf8 -m benchmarks.transfer.smoke_scene8_isic2016
```

It uses the public `stable_pretraining.methods.SimCLR` class, a fresh small
convolutional encoder, two independent 64-pixel augmented views, eight SGD steps
with batch size eight, and only fitting images. W&B is disabled. The executed
`training_smoke.json` records the dependency revision, augmentations, sample IDs,
losses, finite gradients, and actual encoder parameter changes for both datasets.
A short execution check cannot establish meaningful SSL convergence.

The first Windows sandbox execution completed optimization but emitted font-cache,
console-encoding, and temporary-directory cleanup errors in dependencies. A UTF-8
rerun with normal local permissions and a project-local plotting cache completed
cleanly. This environment issue did not require relaxing numerical checks.

## Offline verification

```console
python -m pytest stable_datasets/tests/images/test_scene8.py stable_datasets/tests/images/test_isic2016.py stable_datasets/tests/test_scene8_isic2016_protocols.py stable_datasets/tests/test_manifest_probes.py -q
```

The tests use tiny original-format fixtures and synthetic embeddings. They cover
strict label joins, source preservation, grouped partitions, nested label budgets,
validation-label accounting, tail batches, feature/prediction alignment, known
metric values, deterministic neighbor ties, cache invalidation, and the complete
matrix's refit/selection behavior without downloading an encoder or dataset.
