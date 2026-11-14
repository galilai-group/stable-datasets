# Small real-EDF experiment

The follow-up neonatal correctness experiment and source compatibility findings
are documented in [NEONATAL_EDF_VALIDATION.md](NEONATAL_EDF_VALIDATION.md).

This isolated experiment extends `signal_loading.py`; it changes no builders,
cache formats or public APIs. It uses MNE as the physical-signal reference and
pyEDFlib as an existing CPU EDF reader exposing digital samples. It does not
implement a GPU EDF parser.

## Source and scope

One public recording: PhysioNet EEG Motor Movement/Imagery Dataset, version
1.0.0, subject S001, run R01 (eyes-open baseline).

- Dataset: https://physionet.org/content/eegmmidb/1.0.0/
- Citation: Schalk, G. (2009), DOI https://doi.org/10.13026/C28G6P.
- File: https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf
- Size: 1,275,936 bytes.
- SHA256: `4743b736131a7e147c150e8b37711029b6cda5e356c4b3e8261a03cdcaaf8b0c`.
- 64 EEG channels, 160 Hz, 9,760 samples/channel, 61 seconds.
- One `T0` annotation: onset 0 s, duration 60.2 s.

The experiment accepts only EDF/EDF+C with equal-length, equal-rate voltage
channels. EDF+D, mixed rates, unknown units and non-int16 digital bounds fail
explicitly. This is healthy-subject EEG, not the neonatal seizure dataset;
it does not establish correctness for all medical formats or EDF variants.

References: [EDF](https://www.edfplus.info/specs/edf.html),
[EDF+](https://www.edfplus.info/specs/edfplus.html),
[MNE reader](https://mne.tools/stable/generated/mne.io.read_raw_edf.html).

## Reproduce

Use an environment containing the repository's dependencies plus MNE and
pyEDFlib. For this run they were installed separately, leaving the active
Anaconda environment unchanged:

```bash
python -m pip install --target /tmp/stable-datasets-edf-deps --no-deps mne==1.13.2 pyedflib==0.1.42 pooch==1.9.0
mkdir -p /tmp/stable-datasets-edf-data
curl --fail --location --max-time 90 --output /tmp/stable-datasets-edf-data/S001R01.edf https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf
sha256sum /tmp/stable-datasets-edf-data/S001R01.edf
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.edf_signal_loading /tmp/stable-datasets-edf-data/S001R01.edf --device cuda --repeats 50 --output signal-loading-edf-cuda.json
```

`--no-deps` relies on the other MNE dependencies already present on this machine;
in a fresh environment install MNE/pyEDFlib with dependencies normally.
The `/tmp` installations and source file are temporary and may need recreating.
The output path must not already exist; choose another name for a repeat run.
Run CUDA commands in a terminal with GPU access; the agent sandbox hides this GPU.

## Code structure and refactoring status

`main()` only coordinates configuration, preparation, validated benchmarks and
report output. Internal data contracts are explicit:

- `ExperimentConfig`: CLI paths, device and measurement settings.
- `RecordingMetadata`: channel headers, sample clock and annotations.
- `DigitalRecording`: time-major int16 samples and per-channel voltage calibration.
- `PreparedExperiment`: reference arrays and shared CPU/Arrow/window inputs.

`read_digital()` now returns a `DigitalRecording` rather than a positional tuple;
its experiment tests have been updated accordingly. Metadata is converted to
the original JSON dictionary only at report serialization.

CPU/GPU operation builders name each measured operation explicitly. The shared
timing helper is now `measure_stage()` in `signal_loading.py`; the synthetic
benchmark uses it too. Timing synchronization, stage names, CLI options, units,
tolerances and JSON keys retain their previous meaning. Function-call overhead
can differ after refactoring, so the historical timings below are not new
measurements of this version.

Mandatory metadata checks raise explicit exceptions and remain active with
`python -O`. Full-record windows now keep the forced third start within bounds;
the original implementation could select an invalid start in that case.
Report creation uses exclusive mode to preserve any existing output even if
it appears after argument validation.

Regression cases cover CPU/CUDA execution, full-record windows, JSON
serialization, output preservation and reference metadata mismatch.

Post-refactoring validation on 2026-11-14:

- Both test modules: **28 passed in 3.68 s**, with no skipped CUDA tests.
- Ruff lint/format and `git diff --check`: passed.
- Real EDF CUDA benchmark: passed, 50 repeats; results in
  `signal-loading-edf-cuda-refactored.json` at repository root. Same command as
  above, with this new output filename to preserve the original report.
- Comparison with the original JSON: all non-timing fields match exactly,
  including metadata, window starts and correctness errors. All 11 stage names
  and their result keys match as well.
- Synthetic CUDA CLI smoke run: passed with `--batch-size 3 --length 128
  --channels 2 --repeats 3 --warmup 1`; this verifies the shared timing helper.

In the refactored EDF run, cached float32 gathering/pinning/transfer took a
median 1.827 ms versus 1.075 ms for int16 gathering/pinning/transfer/calibration.
These are separate-run observations, not proof of performance equivalence.
For example, standalone PyTorch CPU calibration changed from 5.152 to 7.682 ms;
benchmark variability and refactoring overhead have not been isolated.

## Observed results before refactoring, 2026-11-14

Base revision `65bf7dcb5cbf214d347c78dee3639bf3e2fb8047`, branch
`research/gpu-medical-timeseries`. The experiment and its new tests/documentation
are uncommitted additions. Existing synthetic JSON files were preserved.

Environment: Intel Core i7-13620H, RTX 4060 Laptop GPU (8 GB), NVIDIA driver
590.48.01, Linux 6.8.0-38-generic/glibc 2.39. Python 3.13.9,
PyTorch 2.15.0+cu132, NumPy 2.3.5, PyArrow 26.0.1, MNE 1.13.2,
pyEDFlib 0.1.42. One PyTorch CPU thread; 3 warmups, 50 repeats.

Validation: 23 tests passed in 3.67 s, including the existing CUDA calibration
test and four new EDF checks (known calibration/metadata, mixed-rate rejection,
unknown-unit rejection and discontinuous EDF rejection). Ruff lint and format
checks passed. A CPU smoke run with 3 repeats also passed before the CUDA run.

All full-record CPU/CUDA samples matched MNE float64 values in volts with
`rtol=2e-6, atol=1e-10 V`. Maximum error for both: approximately
`2.96e-11 V` (0.030 nV). Channel names/order/types, sampling rate, sample count,
recording start and annotation onset/duration/description also matched.
Window checks include recording boundaries and a window crossing a record boundary.

Batch: 32 x 1024 x 64, 8 MiB float32 / 4 MiB int16. Windows are sampled from
this short recording and overlap; these are not 32 independent recordings.

| Stage | Median ms | p95 ms |
|---|---:|---:|
| Full-record MNE preparation to float32 | 4.635 | 6.662 |
| Full-record pyEDFlib preparation to int16 | 24.290 | 24.755 |
| Cached NumPy float32 windows to CPU | 0.507 | 0.561 |
| Cached Arrow float32 windows to CPU | 0.758 | 0.934 |
| NumPy CPU calibration of gathered batch | 1.216 | 1.708 |
| PyTorch CPU calibration of gathered batch | 5.152 | 5.999 |
| NumPy calibration + pinning + GPU transfer | 2.793 | 4.497 |
| PyTorch calibration + pinning + GPU transfer | 4.125 | 5.806 |
| Int16 pinning + GPU transfer + calibration | 0.778 | 1.174 |
| Cached float32 gather + pinning + GPU transfer | 1.906 | 3.421 |
| Cached int16 gather + pinning + GPU transfer + calibration | 1.071 | 1.578 |

Machine-readable results: `signal-loading-edf-cuda.json` at repository root.

## Interpretation and next experiment

For repeated windows, integer gathering/transfer/GPU calibration was about 1.78x
faster than gathering/transferring already calibrated float32. For an already
gathered integer batch it was about 3.59x faster than NumPy calibration plus
pinning/transfer. NumPy was a stronger CPU baseline than the current PyTorch
calibration. Cached NumPy window access also beat the nested Arrow prototype.

The pyEDFlib preparation wrapper was slower than MNE here. It includes metadata
reading, validation and layout conversion and returns a different representation;
these figures do not isolate parser speed. Preparation measurements use a warm
OS file cache; they exclude network download and cold-storage performance.
Batch measurements include output allocation and (where named) per-batch
pinning/gathering. CUDA is synchronized; no model or overlapping pipeline runs.
There is no evidence yet of faster native EDF parsing or faster training.

Next: repeat with representative neonatal EEG metadata/data and an actual
training consumer, comparing persistent float32 and int16 caches with reused
pinned buffers, prefetching and overlap. Include stronger CPU/thread settings,
multiple runs and model GPU contention before proposing library integration.
