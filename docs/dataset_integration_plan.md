# Dataset Integration Plan: Sign Language Digits and FSDD

**Status: Local implementation and validation are complete. The user-owned commits and remote PR update remain pending.**

## Overview

This contribution proposes adding two classification datasets to `stable-datasets`:

1. **Sign Language Digits**, from `ardamavi/Sign-Language-Digits-Dataset`.
2. **Free Spoken Digit Dataset (FSDD)**, from `Jakobovski/free-spoken-digit-dataset`.

The contribution will include dataset builders, public exports, tests, documentation, and reproducible training and evaluation examples using existing supervised methods.

## 1. Sign Language Digits

### Source

- Repository: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- Planned source revision: `4b09433ef9d5c60602e8805bb54b8aa2428527f8`
- License: Apache-2.0.
- Task: Classification of RGB hand-sign images representing digits 0–9.
- This dataset is distinct from Sign Language MNIST.

### Implementation

- Add a `SignLanguageDigits` builder under `stable_datasets/images/`.
- Read images only from `Dataset/<digit>/`.
- Exclude preview images in `Examples/` and unrelated files.
- Use a fixed class mapping from digit names to labels 0–9.
- Preserve stable filenames for sample identification.
- Expose the original image pool as an `all` split.
- Document that the source does not provide an official train/validation/test partition.

### Evaluation split

Create a reproducible, class-stratified train/validation/test partition in the experiment setup, using a fixed seed and saved sample manifests.

The proposed proportions are approximately 80% training, 10% validation, and 10% testing. These will be described as experiment-defined splits. No signer-independent evaluation will be claimed without reliable signer identifiers.

## 2. Free Spoken Digit Dataset (FSDD)

### Source

- Repository: https://github.com/Jakobovski/free-spoken-digit-dataset
- Planned source revision: `26eb9aaf76e81b692f806f9140c2d2777410d7a1`
- License: Creative Commons Attribution-ShareAlike 4.0 International.
- Task: Classification of spoken English digits 0–9.
- This dataset is distinct from AudioMNIST.

### Implementation

- Add an `FSDD` builder under `stable_datasets/timeseries/`.
- Read WAV files from the `recordings/` directory.
- Parse filenames using the `{digit}_{speaker}_{index}.wav` convention.
- Preserve the digit label, speaker identifier, recording index, filename, and actual sample rate.
- Validate sample rates, channel counts, and non-empty audio.
- Document the waveform layout and amplitude convention.
- Preserve the official training and test splits:
  - Recording indices 0–4: test.
  - Recording indices 5–49: train.

### Evaluation split

Create a validation subset from the official training split. Keep the official test set separate from model selection.

The evaluation will not be described as speaker-independent because the official training and test partitions contain the same speakers.

## 3. Tests and Data Validation

Tests will use small local image and WAV fixtures packaged into archives, with downloads mocked where appropriate.

Planned coverage includes:

- Public imports and dataset metadata.
- Archive parsing and exclusion of unrelated files.
- Correct class labels and stable sample identifiers.
- Split boundaries and non-overlapping partitions.
- Image decoding and RGB handling.
- Audio decoding, sample rates, waveform layout, and amplitude handling.
- Cache reuse.
- Clear handling of invalid inputs.
- Batching and a small training integration check, including variable-length audio.

Both datasets will also be downloaded and decoded in full to verify actual sample counts, class distributions, and data integrity.

## 4. Training and Evaluation

Use lightweight supervised CNN baselines through `stable-pretraining`:

- A small 2D CNN for Sign Language Digits.
- A small 1D CNN for FSDD, with appropriate padding and valid-length handling.

Both experiments will load data through the new `stable-datasets` builders.

The experiment records will include:

- Dataset and repository revisions.
- Split definitions and random seeds.
- Model architecture and preprocessing.
- Optimizer, learning rate, batch size, and actual training budget.
- Training and validation loss and accuracy.
- Final test results after model selection is complete.
- Software versions, hardware, execution commands, and runtime.

An initial smoke test will check the training pipeline. Separate training runs will produce the reported validation results.

## 5. Completion Checklist

- [x] Implement the Sign Language Digits builder.
- [x] Implement the FSDD builder.
- [x] Add public exports for both datasets.
- [x] Add and pass local-fixture tests.
- [x] Verify actual downloads, decoding, labels, and sample counts.
- [x] Implement reproducible training and evaluation examples.
- [x] Run supervised validation on both datasets.
- [x] Record actual metrics and reproduction instructions.
- [x] Add dataset documentation and documentation index entries.
- [x] Run local formatting checks and review the currently gated CI results.
- [ ] Update the PR description with completed work and remaining limitations.

The final item is intentionally left open because the course requires the user,
not an agent, to commit, push, and update the remote PR. A ready-to-paste PR
description is included in the local handoff.
