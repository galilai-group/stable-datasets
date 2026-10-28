# Dataset Integration Status: Sign Language Digits and FSDD

**Status:** The builders, exports, fixture tests, dataset documentation, validation script,
and two 15-epoch reference runs are committed on `add-sign-digits-fsdd` at
`34b02e1f94ac3bab4c25183f257fae297cb80469`. This working tree adds a follow-up
image-normalization fix, regression coverage, and documentation corrections that have
not yet been committed or pushed by the contributor.

## Overview

This contribution adds two digit-classification datasets:

1. **Sign Language Digits**, from `ardamavi/Sign-Language-Digits-Dataset`.
2. **Free Spoken Digit Dataset (FSDD)**, from `Jakobovski/free-spoken-digit-dataset`.

The committed contribution includes dataset builders, public exports, tests,
documentation, and reproducible supervised training and evaluation examples using
`stable-pretraining`.

## 1. Sign Language Digits

### Source and builder

- Repository: https://github.com/ardamavi/Sign-Language-Digits-Dataset
- Pinned source revision: `4b09433ef9d5c60602e8805bb54b8aa2428527f8`
- License: Apache-2.0.
- Task: classification of RGB hand-sign images representing digits 0--9.
- The builder reads images only from `Dataset/<digit>/`, excludes previews and unrelated
  files, uses a fixed digit-to-label mapping, preserves stable logical filenames, and
  exposes the source image pool as the `all` split.
- The source does not provide an official train, validation, or test partition. This
  dataset is distinct from Sign Language MNIST.

### Evaluation split

The reference experiment uses a deterministic, class-stratified split with seed 42 and
tracked manifests: 1,648 train, 210 validation, and 204 test images. These are explicitly
experiment-defined splits. They are not presented as signer-independent because the
source does not provide reliable signer identifiers for that purpose.

## 2. Free Spoken Digit Dataset (FSDD)

### Source and builder

- Repository: https://github.com/Jakobovski/free-spoken-digit-dataset
- Pinned source revision: `26eb9aaf76e81b692f806f9140c2d2777410d7a1`
- License: Creative Commons Attribution-ShareAlike 4.0 International.
- Task: classification of spoken English digits 0--9. This dataset is distinct from
  AudioMNIST.
- The builder reads WAV files directly below a `recordings/` directory and parses the
  `{digit}_{speaker}_{index}.wav` convention.
- Each example preserves the label, speaker identifier, utterance index, filename,
  sample rate, and a float32 column vector containing the original signed 16-bit PCM
  values.
- The builder validates filenames, 8 kHz sample rate, mono layout, non-empty and finite
  samples, signed 16-bit PCM dtype, duplicate logical filenames, and archive structure.
- It preserves the official split: indices 0--4 are test and indices 5--49 are train.

### Evaluation split

The reference experiment reserves indices 5--9 from the official training split for
validation, uses indices 10--49 for fitting, and evaluates the official indices 0--4 test
set only after checkpoint selection. The resulting counts are 2,400 train, 300
validation, and 300 test recordings. The protocol is not speaker-independent because
the same six speakers occur in every subset.

## 3. Tests and Data Validation

The committed tests use small local image and WAV archives with downloads mocked where
appropriate. They cover public imports and metadata, archive filtering, labels and
stable identifiers, official and experiment split boundaries, image and audio decoding,
cache reuse, malformed inputs, variable-length batching, forward/backward updates, and
a one-epoch integration run through the real `stable_pretraining.Manager` when its
optional dependencies are installed.

Both pinned archives were also decoded in full. The recorded checks found 2,062 valid
Sign Language Digits images and 3,000 valid FSDD recordings with the documented counts,
formats, and distributions.

The current uncommitted follow-up adds regression tests for low-dynamic-range uint8 image
normalization, explicit normalized-float handling, duplicate logical filenames under
different archive roots, and finite WAV data that is not signed 16-bit PCM.

The latest targeted run completed with 41 passed, 0 failed, and 0 skipped tests. Its
one-epoch `stable_pretraining.Manager` integration test executed and passed; the six
reported warnings came from Lightning or `stable-pretraining` rather than skipped tests.

## 4. Training and Evaluation

The validation script trains deliberately small supervised baselines through
`stable-pretraining`: a 2D CNN for Sign Language Digits and a padding-aware 1D CNN for
FSDD. Both runs used CPU, seed 42, AdamW, batch size 64, and 15 epochs. Validation
accuracy selected the checkpoint before one final test evaluation.

| Dataset | Train / validation / test | Best epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Sign Language Digits | 1,648 / 210 / 204 | 10 | 94.7619% | 94.6078% |
| FSDD | 2,400 / 300 / 300 | 15 | 44.0000% | 47.0000% |

The FSDD result is a weak baseline and is reported without post-hoc tuning on the test
set. The train accuracy stored for each selected epoch is the aggregate observed during
that training epoch, not a separate evaluation of the restored best checkpoint on the
training set.

The tracked `metrics.json`, `metrics.csv`, and manifests contain the available historical
metadata. They do not record a `stable-datasets` Git SHA or dirty state, so no such value
is backfilled after the fact. The exact recorded `stable-pretraining` revision is
`ab836bf699a2be712dfcf980a5eb70d36391e876`.

## 5. Completion and Remaining Work

### Committed at `34b02e1`

- [x] Implement and publicly export both builders.
- [x] Add local-fixture tests and builder/training integration coverage.
- [x] Validate the pinned real datasets and record their manifests and statistics.
- [x] Add the supervised validation script and run both 15-epoch CPU experiments.
- [x] Track metrics, split manifests, dataset documentation, and reproduction guidance.

### Current uncommitted follow-up

- [x] Correct PIL/uint8 normalization without inferring scale from a sample maximum.
- [x] Add normalization and optional builder edge-case regression tests.
- [x] Pin the reference `stable-pretraining` revision in reproduction instructions.
- [x] Complete and record the final targeted pytest, formatting, lint, and documentation
  checks.

All configured pre-commit hooks applicable to the eight changed files passed. The
repository's configured non-strict Sphinx build also completed, with 31 warnings in
pre-existing documentation or configuration. A stricter `-W --keep-going` build treats
the same warnings as errors; they include the absent root `CONTRIBUTING.md`, missing
`_static` assets, older title underlines, toctree omissions, and an unresolved `mnist`
reference. Neither changed dataset RST file produced a warning.

### Contributor and remote actions still pending

- [ ] Review, commit, and push the current working-tree changes as the contributor.
- [ ] Replace the placeholder PR body with the completed implementation, test, experiment,
  and limitation details.
- [ ] Remove `[WIP]` and mark the PR ready for review when appropriate.
- [ ] Obtain any required approval for fork-originated workflows and confirm that remote
  CI jobs actually run.

At the 2026-10-23 check of PR head `34b02e1`, the Testing and pre-commit workflow
runs both ended with GitHub's `action_required` conclusion and exposed no jobs or check
runs. This is not a passing result, but it is also not evidence of a code or test failure.
