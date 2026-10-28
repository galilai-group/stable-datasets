# Digit validation results

These lightweight artifacts record the reportable CPU runs for the two digit
datasets. Both experiments used seed 42, 15 epochs, batch size 64, AdamW with
learning rate `1e-3` and weight decay `1e-4`, and a step-wise cosine schedule.
The exact commands, package versions, runtime, complete epoch histories, split
configuration, and preprocessing are captured in each `metrics.json`.

| Dataset | Train / validation / test | Best epoch | Validation accuracy | Test accuracy | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sign Language Digits | 1,648 / 210 / 204 | 10 | 94.7619% | 94.6078% | 88.37 s |
| FSDD | 2,400 / 300 / 300 | 15 | 44.0000% | 47.0000% | 90.34 s |

These are small supervised CNN baselines rather than self-supervised or
state-of-the-art results. In particular, FSDD's 47.00% test accuracy is a weak
baseline and is reported without post-hoc test-set tuning. The train metrics
stored for the selected epoch are training-loop epoch aggregates, not a
separate evaluation of the restored best checkpoint on the training split.

## Reproducible environment

The `benchmarks` extra in `pyproject.toml` follows the current
`stable-pretraining` Git source without a revision pin. The reference artifacts
record Python 3.12.6 and `stable-pretraining` version
`0.1.9.dev10+gab836bf69`; retained VCS installation metadata resolves it to
commit `ab836bf699a2be712dfcf980a5eb70d36391e876`. The same retained environment
verifies the core package versions below, including torchvision 0.29.1:

```console
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e . \
    "stable-pretraining @ git+https://github.com/galilai-group/stable-pretraining.git@ab836bf699a2be712dfcf980a5eb70d36391e876" \
    "torch==2.14.1" "torchvision==0.29.1" "lightning==2.6.6" \
    "numpy==2.5.4" "pillow==13.0.0"
```

Run the two reference configurations from the repository root:

```console
python examples/validate_digits.py \
    --dataset sign-language-digits --epochs 15 --batch-size 64 \
    --learning-rate 0.001 --weight-decay 0.0001 --seed 42 --device cpu \
    --output-dir benchmark-runs/digit-validation

python examples/validate_digits.py \
    --dataset fsdd --epochs 15 --batch-size 64 \
    --learning-rate 0.001 --weight-decay 0.0001 --seed 42 --device cpu \
    --output-dir benchmark-runs/digit-validation
```

Each dataset directory contains the exact `metrics.json`, long-form
`metrics.csv`, and deterministic train/validation/test manifests emitted by
`examples/validate_digits.py`. The best-validation Lightning checkpoints were
used to compute the reported test metrics but are intentionally excluded from
this tracked results directory; rerunning the command recorded in a metrics
file recreates its checkpoint under `benchmark-runs/digit-validation/`.

The pinned source archives were also decoded in full before training. Sign
Language Digits contained 2,062 unique RGB images with class counts
`[205, 206, 206, 206, 207, 207, 207, 206, 208, 204]`; 2,059 images were
100 x 100 and three class-7 images were 3024 x 3024. No decode failures or
pixel-identical duplicates were found. FSDD contained 2,700 official-train and
300 official-test recordings, with balanced classes and six speakers. Every
recording was finite mono 8 kHz signed-16-bit PCM. Train lengths ranged from
1,149 to 18,262 samples (median 3,357), test lengths from 1,148 to 9,178
(median 3,362), and no exact waveform duplicate crossed the split boundary.

A follow-up check of all 2,062 Sign Language Digits inputs found that their
per-image maxima after the experiment's RGB conversion and 64 x 64 resize were
between 174 and 255. The low-dynamic-range uint8 normalization regression fix
therefore leaves every historical experiment input unchanged, so these
artifacts were not regenerated. The historical files do not contain a
`stable-datasets` Git SHA or dirty-state field, and no unavailable provenance
has been backfilled.
