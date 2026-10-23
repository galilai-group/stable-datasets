# ETH-80 and Freiburg Groceries: measured results

Hardware: Apple M4 Mac mini, 16 GB memory, four CPU threads. Software: Python 3.12.14 and PyTorch 2.14.1. Both encoder conditions use random initialization; the LeJEPA condition receives SSL training and the random baseline encoder remains untrained.

The benchmark uses 20 SSL epochs per dataset and seeds 42, 43, 44. Training settings were fixed before test evaluation. Both encoder conditions use identical dataset partitions and the same frozen linear-probe selection protocol. [Training and evaluation protocol](object-datasets-validation.md).

## Test accuracy

Values are percentages, mean ± sample standard deviation across three training seeds. Gain is the mean paired difference in percentage points. Results use the protocol specified below; the dataset partitions and training configuration differ from published benchmark protocols.

| Dataset | Random encoder | LeJEPA encoder | Paired gain (pp) |
| --- | ---: | ---: | ---: |
| ETH-80 | 54.88 ± 2.78 | 56.81 ± 2.66 | +1.93 |
| Freiburg Groceries | 19.19 ± 0.74 | 28.42 ± 0.91 | +9.23 |

## Per-seed results

| Dataset | Seed | Encoder | Correct / test | Top-1 (%) | Balanced (%) | Validation (%) | Probe iteration | Weight decay |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH-80 | 42 | random | 348 / 656 | 53.05 | 53.05 | 66.31 | 36 | 0 |
| ETH-80 | 42 | lejepa | 368 / 656 | 56.10 | 56.10 | 69.66 | 64 | 0 |
| ETH-80 | 43 | random | 381 / 656 | 58.08 | 58.08 | 67.68 | 189 | 0 |
| ETH-80 | 43 | lejepa | 392 / 656 | 59.76 | 59.76 | 71.65 | 200 | 0 |
| ETH-80 | 44 | random | 351 / 656 | 53.51 | 53.51 | 66.01 | 76 | 0 |
| ETH-80 | 44 | lejepa | 358 / 656 | 54.57 | 54.57 | 70.12 | 134 | 0 |
| Freiburg Groceries | 42 | random | 204 / 1018 | 20.04 | 18.91 | 21.25 | 121 | 0 |
| Freiburg Groceries | 42 | lejepa | 284 / 1018 | 27.90 | 26.61 | 27.86 | 195 | 0 |
| Freiburg Groceries | 43 | random | 190 / 1018 | 18.66 | 17.76 | 18.58 | 96 | 0 |
| Freiburg Groceries | 43 | lejepa | 300 / 1018 | 29.47 | 27.42 | 27.99 | 198 | 0 |
| Freiburg Groceries | 44 | random | 192 / 1018 | 18.86 | 17.93 | 19.59 | 61 | 0 |
| Freiburg Groceries | 44 | lejepa | 284 / 1018 | 27.90 | 26.28 | 27.74 | 191 | 0 |

## Training and interpretation

- ETH-80: SSL took 2.66–2.83 minutes per seed, excluding probe evaluation. Mean test change was +1.93 percentage points; LeJEPA exceeded the paired random encoder in 3/3 seeds.
- Freiburg Groceries: SSL took 4.26–4.39 minutes per seed, excluding probe evaluation. Mean test change was +9.23 percentage points; LeJEPA exceeded the paired random encoder in 3/3 seeds.

The protocol uses 64-pixel global inputs, a 192 → 256 → 128 projection head, 64 SIGReg slices and 20 epochs. Reported variability covers initialization and training on a fixed partition. Alternative physical-object partitions and Freiburg folds were not evaluated.

ETH-80 evaluates unseen objects with a custom instance-level split, which differs from leave-one-object-out evaluation. Freiburg preserves the official test set and uses a deduplicated training subset with an additional validation split. Physical product identities are unavailable in Freiburg; path separation does not establish physical-instance separation.

## Verified data and checks

- ETH-80: all 3,280 released images decoded; train/validation/test contain 1,968/656/656 images and 48/16/16 disjoint objects. Non-contiguous car IDs are preserved.
- Freiburg: all 4,947 released images decoded; the default loader preserves 7,425 training rows (3,929 unique paths) and 1,018 test images. All five author manifests were audited for path/label agreement, source existence and train/test separation.
- Every cached sample was compared against source RGB pixels; processed caches reopened identically, and both datasets formed [8, 3, 256, 256] batches.
- Test coverage: 20 dataset/benchmark offline tests and 139 builder/cache/dataset/schema regressions; all 159 passed on macOS with Python 3.10.22 and Python 3.12.14.
- Static checks: configured pre-commit hooks and Ruff passed. The Slurm script passed ShellCheck and Bash syntax validation.
- Documentation: Sphinx built successfully with 31 pre-existing diagnostics, including the missing CONTRIBUTING.md include. The dataset pages and teaser images introduce no additional diagnostics.
- Execution coverage: training and evaluation ran on CPU with Python 3.12.14. CUDA/Oscar execution and the full repository dataset-download test suite are outside the recorded validation scope.

## Saved artifacts and limitations

The experiment writes config.json, environment.json, requirements-lock.txt, splits.json, training.json, checkpoint.pt, random-encoder.pt, selected linear heads, validation-selection traces, results.json, and test-prediction CSVs. Reproduction commands are in the protocol linked above. Full source archives and model checkpoints are excluded from the Git contribution.

No explicit dataset license was found in the referenced ETH-80 provider page or cropped256 archive. The library's software license does not grant image rights; image usage remains governed by the original provider. Freiburg's author page restricts use to research, requires consent for commercial use, and requests citation. The documentation contains five attributed previews per dataset under the original image terms. Full source archives are not included.
