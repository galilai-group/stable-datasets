# Neonatal EDF correctness experiment

The subsequent paired-cache implementation and training experiment are described
in [SIGNAL_CACHE_TRAINING.md](SIGNAL_CACHE_TRAINING.md).

## Purpose and source

Validate selected windows of a representative medical EEG recording against
the unmodified MNE source. This step does not measure throughput or training.

Use only `eeg4.edf` (36,831,232 bytes) and `annotations_2017.mat` (12,889 bytes)
from the exact version used by `SeizuresNeonatal`:

- [Zenodo record 2547147, v3](https://zenodo.org/records/2547147), CC BY 4.0.
- Stevenson et al., [Scientific Data (2019)](https://www.nature.com/articles/sdata201939).
- Source EDF MD5: `af2f439c279204c1a61849a91c297894`.
- Annotations MD5: `f0009a921d9dacd135fff4fc2377d872`.
- Source EDF SHA256: `6444722cf46e101381754354888194dc1402ad445cd3da3df1df212e6365c81e`.
- Annotations SHA256: `a82a2a31e8f03565cb065c373c56901aedf4931591bf12dd3a03dea04b511ac6`.

Published MD5 checksums matched both downloaded files. The full dataset was not
downloaded. No clinical/demographic table was needed.

## Compatibility findings

MNE 1.13.2 reads the original file, while pyEDFlib 0.1.42 rejects it as
noncompliant (`the label is incorrect`). Inspection found an `EDF+C` marker
without the EDF Annotations channel required by EDF+. Clearing only the fixed
header's reserved field (bytes 192:236) in a separate ordinary-EDF derivative
allows pyEDFlib to read it. This is an experiment-specific workaround, not proof
that the original file is supported by our strict reader or that arbitrary
EDF+C files can safely be relabelled.

`normalize_continuous_header()` verifies that every byte outside that field is
unchanged, including all channel metadata and signal bytes. It refuses sources
with an EDF Annotations channel and never overwrites an existing file. The
original remains the MNE reference. Derivative SHA256:
`109a4ea386076e2cb99e8afddfda27961eac3260c5996f64e5d3ed378e2ed4c2`.

A second discrepancy: digital bounds are declared as -32767..32767, but 1,585
samples contain -32768 (1 in EEG F3-Ref, 1,584 in ECG EKG). The standard reader
still rejects these values by default. An explicit
`allow_out_of_range_samples=True` retains them unchanged for this investigation;
it neither clips samples nor changes calibration bounds. Representability as
int16 is always enforced. Counts are included in the result JSON, and windows
containing each affected channel's first out-of-range sample are checked.

## Procedure and code

`profiling/neonatal_edf_validation.py` separates source normalization, external
label loading, deterministic window selection, numerical checks and reporting.
It reuses the existing typed recording/metadata and calibration helpers.

Preparation reads the full integer recording once via pyEDFlib. MNE uses
`preload=False` and reads selected reference windows only. This is not a native
integer window-reader or performance benchmark.

The MAT schema is validated as three independent binary expert rows at 1 Hz,
indexed by one-based subject ID. Filename and ID must match on the CLI. Labels
are not majority-voted or transposed. A sample window `[start, stop)` intersects
annotation seconds `[floor(start/fs), floor((stop-1)/fs)+1)`. The experiment
checks coverage and sample/time indexing using the publication's alignment
convention; it does not independently validate clinical seizure labels.

## Reproduce

Use the temporary dependency setup documented in `EDF_SIGNAL_LOADING.md`.
Source, derivative and output paths must be distinct. Choose new derivative
and JSON names when repeating; existing files are preserved.

```bash
mkdir -p /tmp/stable-datasets-neonatal
curl --fail --location --max-time 90 --output /tmp/stable-datasets-neonatal/eeg4.edf https://zenodo.org/records/2547147/files/eeg4.edf
curl --fail --location --max-time 30 --output /tmp/stable-datasets-neonatal/annotations_2017.mat https://zenodo.org/records/2547147/files/annotations_2017.mat
md5sum /tmp/stable-datasets-neonatal/eeg4.edf /tmp/stable-datasets-neonatal/annotations_2017.mat
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.neonatal_edf_validation /tmp/stable-datasets-neonatal/eeg4.edf --annotations /tmp/stable-datasets-neonatal/annotations_2017.mat --subject-id 4 --device cuda --normalized-copy /tmp/stable-datasets-neonatal/eeg4-validation-copy.edf --allow-out-of-range-samples --output signal-loading-neonatal-validation.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py stable_datasets/tests/test_neonatal_edf_validation.py
```

Run CUDA outside the agent sandbox. No dataset downloads occur during tests;
the new regression tests construct small EDF fixtures locally.

## Observed results, 2026-11-14

Base revision: `65bf7dcb5cbf214d347c78dee3639bf3e2fb8047`, branch
`research/gpu-medical-timeseries`, with uncommitted experimental additions.
Hardware/environment are the same as the preceding EDF experiment: RTX 4060
Laptop GPU, Python 3.13.9, PyTorch 2.15.0+cu132, NumPy 2.3.5, MNE 1.13.2,
pyEDFlib 0.1.42. Additional library versions are recorded in the JSON.

- 21 retained channels: 19 EEG-labelled, ECG EKG and Resp Effort.
- Every channel is declared in uV at 256 Hz, with 876,800 samples (3,425 s).
- The builder's default MNE reader classifies all 21 as EEG. This experiment
  preserves its behavior, but does not interpret ECG/respiration as EEG signals.
- Header date/time are anonymized dataset fields, not a physiological timestamp.
- No embedded EDF annotations. The external MAT array has shape `(3, 3425)`
  and exactly matches recording duration. Positive seconds by expert:
  925, 1,161 and 987; these are not seizure-event counts or consensus labels.
- Twelve windows of 1,024 samples (4 seconds) passed on CPU and CUDA, including
  first/last windows, the midpoint, second/annotation boundaries, and anomalies.
- Maximum absolute CPU and CUDA error against MNE float64: approximately
  `6.10e-10 V` (0.61 nV). All samples satisfy the existing combined tolerance
  `abs(error) <= 1e-10 V + 2e-6 * abs(reference)`; this is not an assertion that
  every absolute error is below `1e-10 V`.
- Metadata comparison passed for channel order, rate, length, anonymized start,
  and empty embedded annotations.
- **34 tests passed in 4.04 s**, with no skips, including CUDA. New tests cover
  exact derivative byte preservation, rejection of annotation-bearing sources,
  strict versus opt-in digital bounds, CPU/CUDA window/label alignment,
  annotation coverage mismatch and full-record window bounds.

Two exploratory attempts failed before the successful comparison: direct
pyEDFlib reading of the original, then strict sample-bound validation of the
header-normalized copy. These failures are findings, not discarded test failures.
The small CPU check and final CPU/CUDA check both passed with explicit handling.
All source/derivative data stays under `/tmp`; the report is
`signal-loading-neonatal-validation.json` at repository root. No Git changes
were staged, committed or pushed.

## Next decision

Calibration works on the selected neonatal windows, but original-source
compatibility requires explicit handling. Before a persistent int16 cache or
training comparison, define whether MNE remains the authoritative preparation
reader or whether these inconsistencies can be supported safely and explicitly.
Then compare identical windows in persistent float32/int16 caches with an actual
training consumer, including preparation cost and model/GPU contention.
