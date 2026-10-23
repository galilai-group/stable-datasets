# Digit validation results

These lightweight artifacts record the reportable CPU runs for the two digit
datasets. Both experiments used seed 42, 15 epochs, batch size 64, AdamW with
learning rate `1e-3` and weight decay `1e-4`, and a step-wise cosine schedule.
The exact commands, package versions, runtime, complete epoch histories, split
configuration, and preprocessing are captured in each `metrics.json`.

| Dataset | Best epoch | Validation accuracy | Test accuracy | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Sign Language Digits | 10 | 94.76% | 94.61% | 88.37 s |
| FSDD | 15 | 44.00% | 47.00% | 90.34 s |

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
