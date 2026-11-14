# Native WFDB-212: compact CPU/GPU decoding experiment

## Question and scope

Can original packed physiological-signal bytes be decoded on GPU faster than a
strong CPU path, including source-window gathering and transfers? This addresses
Sami's native-format/GPU-decoding question, separately from earlier cache work.

The supported subset is explicit: a single-segment record, two synchronized
channels, one sample per channel/frame, one shared format-212 file and common
byte offset. Skew, multi-segment/multi-file layouts, unsupported units and other
formats fail. There is no resampling, source rewriting or production loader.

## Original source and primary references

One public original record, MIT-BIH Arrhythmia Database 1.0.0, record 100:

- [Dataset and citation](https://physionet.org/content/mitdb/1.0.0/),
  Moody and Mark (2005), DOI https://doi.org/10.13026/C2F305.
- [Original header](https://physionet.org/content/mitdb/1.0.0/100.hea).
- Two ECG channels, MLII and V5; 650,000 frames at 360 Hz.
- ADC gains 200 ADU/mV, baseline 1024 on both channels.
- `100.dat`: 1,950,000 bytes; header: 143 bytes.
- SHA256 `100.dat`:
  `b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639`.
- SHA256 `100.hea`:
  `60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e`.

Format 212 stores two signed 12-bit samples in three bytes. Multiplexed sample
order must be retained. The raw -2048 value denotes invalid/missing data;
physical outputs preserve this as NaN, not a calibrated waveform amplitude.
WFDB Python returns the raw sentinel in digital mode; the C API may map invalid
data to its API-level invalid-sample constant. Our exact digital comparison is
against the Python reader's raw convention.

Specifications: [WFDB encodings](https://wfdb.io/spec/signal-files.html),
[precise 212 bit layout](https://physionet.org/physiotools/wag/signal-5.htm),
[invalid sample convention](https://physionet.org/physiotools/wpg/wpg_78.htm),
[WFDB Python reference API](https://wfdb.readthedocs.io/en/latest/io.html).

## Small implementation

- `wfdb212.py`: metadata via WFDB, read-only original byte mapping, contiguous
  window copies, vectorized NumPy unpack/sign extension and calibration to volts.
- `wfdb212_cpu.py`: one short unpack/calibration loop compiled serially and
  with four Numba threads. Both compute final physical values in one pass.
- `wfdb212_cuda.py`: one Triton kernel fuses unpacking, sign extension,
  missing-value handling and calibration. The same kernel also provides an
  int16 calibration-only hybrid control and a digital-only correctness mode.
- `benchmark_wfdb212.py`: independent reference checks, balanced trial order,
  separate prepared-input diagnostics and JSON output.

The NumPy path is not assumed to be the best CPU implementation. Compiled
serial/parallel paths and an int16 CPU-unpack/GPU-calibration control prevent
comparison only against general-reader/Python overhead. CPU/GPU fast-math/FMA
fusion is not enabled. Numba specializations compile separately in memory to
avoid mixing serial/parallel artifacts in a shared function disk cache.

An early CPU JIT check exposed unsigned/signed type inference in a conditional;
explicit int32 byte promotion fixed it before performance results were accepted.
The full-window start boundary was also covered and corrected before final runs.

## Measurement protocol

All six complete paths end with GPU float32 signals in volts and identical NaN
semantics. Timing includes gathering original packed bytes, allocations,
per-batch pinning, H2D transfer and the specified decoding/calibration. Header
parsing, constant GPU metadata and JIT compilation occur before timing.
The WFDB general-reader diagnostic opens/reads each reference window normally;
headline comparisons use the strongest specialized CPU/hybrid alternative.

For each workload, six rounds rotate a seeded permutation of the six paths;
each path occupies every position once. Each round has three warmups and thirty
timed calls, synchronized to completed-batch latency. Reported values are
medians of round medians. A second fresh-process series reverses workload order.
There are 360 timed calls per path/workload across the two final series.

The source is accessed on each call, but OS pages are warm from checks/previous
reads. These are not cold-disk, training or overlapped-pipeline measurements.
GPU source-file opening/header parsing are not implemented. No physical storage
traffic, model contention or system clock/power isolation is claimed.

## Final results, 2026-11-14

Same RTX 4060 Laptop GPU; Python 3.13.9, NumPy 2.3.5,
PyTorch 2.15.0+cu132, WFDB 4.3.1, Numba 0.62.1, Triton 3.9.0. One PyTorch CPU
thread; compiled CPU parallel path explicitly uses four threads.
Base commit: `cd5e3b259baede94fd659696b7d438ee35dfd524`; new work is uncommitted.

Final reports: `profiling/medical_signal_research/results/wfdb212-final-a/`, `profiling/medical_signal_research/results/wfdb212-final-b/` and
`profiling/medical_signal_research/results/wfdb212-final-summary.json`. Source/code signatures and
window requests match between series. Earlier `profiling/medical_signal_research/archive/results/wfdb212-results-v1/` and `v2/`
and `profiling/medical_signal_research/archive/results/wfdb212-summary.json` are preserved exploratory results preceding the
full-window boundary regression; they are not pooled into the final figures.

| Batch x frames x channels | Best CPU/hybrid, A/B ms | Packed GPU, A/B ms | Interpretation |
|---|---:|---:|---|
| 4 x 128 x 2 | 0.0306 / 0.0300 | 0.0650 / 0.0749 | Serial compiled CPU wins |
| 32 x 1024 x 2 | 0.1490 / 0.1590 | 0.1228 / 0.1274 | GPU latency lower by 17.6% / 19.9% |
| 32 x 8192 x 2 | 0.7775 / 0.8483 | 0.2473 / 0.2360 | GPU latency lower by 68.2% / 72.2% |

The strongest medium/large baseline was the four-thread compiled CPU decoder.
The large-batch latency ratio is about 3.1–3.6, not a training-throughput speedup.
The general WFDB reader took approximately 23 ms for that batch; using that
reader alone as baseline would substantially exaggerate the GPU-specific result.

Packed bytes versus physical float32 payloads are 1,536/4,096 bytes for the
small batch, 98,304/262,144 for medium and 786,432/2,097,152 for large. Moving
original packed bytes reduces signal H2D payload by 62.5% compared with CPU
float32 decoding, before small constant metadata. A dedicated prepared-GPU
diagnostic excludes source/pinning/transfer and is not substituted for the
complete-path measurements. Small-batch launch/transfer overhead can outweigh
the saved arithmetic/bytes. Stage timings are not additive causal attribution.

## Correctness

**70 targeted tests passed in 6.91 s**, with no skipped CUDA tests. The twelve
WFDB cases cover every signed 12-bit value, random channel pairs, nonzero
baselines, different units/gains, missing-sample masks, odd starts/lengths,
duplicates, end boundaries, byte offsets, full windows, unsupported headers,
truncation and all complete source-to-GPU paths.

The complete original record was additionally checked: all 1.3 million digital
samples match WFDB bit-for-bit on CPU and CUDA. Every calibrated value/NaN mask
matches the float64 reference within `rtol=2e-6, atol=1e-10 V`. Maximum physical
error was `1.6861e-10 V` (about 0.17 nV). This record has no missing samples;
missing semantics are exercised by synthetic reference-written fixtures.

Full-record artifacts: `profiling/medical_signal_research/results/wfdb212-full-record-validation.json` and the reproducible
all-path `profiling/medical_signal_research/results/wfdb212-full-record-check.json`. The latter's one-repeat timings are
correctness diagnostics only. Header/data hashes remained unchanged throughout.
Ruff lint/format checks passed. No source record, Git index, commit or remote
was modified.

## Reproduce

Install WFDB separately, as done here. Numba/Triton and CUDA-enabled PyTorch
were already present; their exact versions are above. A fresh environment also
needs the normal WFDB dependencies (`--no-deps` assumes the existing environment).

```bash
python -m pip install --target /tmp/stable-datasets-edf-deps --no-deps wfdb==4.3.1
mkdir -p /tmp/stable-datasets-wfdb
curl --fail --location --max-time 60 --output /tmp/stable-datasets-wfdb/100.hea https://physionet.org/files/mitdb/1.0.0/100.hea
curl --fail --location --max-time 60 --output /tmp/stable-datasets-wfdb/100.dat https://physionet.org/files/mitdb/1.0.0/100.dat
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_wfdb212.py
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_wfdb212 /tmp/stable-datasets-wfdb/100 --batch-size 4 --length 128 --output fresh-small.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_wfdb212 /tmp/stable-datasets-wfdb/100 --batch-size 32 --length 1024 --output fresh-medium.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_wfdb212 /tmp/stable-datasets-wfdb/100 --batch-size 32 --length 8192 --output fresh-large.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_wfdb212 /tmp/stable-datasets-wfdb/100 --batch-size 3 --length 650000 --rounds 1 --repeats 1 --output fresh-full-check.json
```

For the second series run the same workloads large/medium/small with fresh
filenames. Default rounds=6 and repeats=30 match the study. CUDA requires access
outside the agent sandbox. Recreate `/tmp` data/dependencies if removed. These
commands download one recording only, not the database or annotations.
