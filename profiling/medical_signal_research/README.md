# Native physiological signals on GPU

This research prototype investigates Sami's request for fast loading or GPU
interpretation of time-series and medical formats. It covers native EDF/EDF+C
record assembly and WFDB-212 packed-sample decoding. Production dataset builders,
cache formats and defaults are unchanged.

## Findings

The retained measurements used two EEG recordings, one ECG recording and one
RTX 4060 Laptop GPU. All paths end at float32 GPU tensors in volts. Completed
batch timings include source-window gathering, allocation, pinning and transfer;
header parsing, constant GPU metadata and JIT compilation are excluded.

| Workload | Best measured CPU/hybrid | GPU | Outcome |
|---|---:|---:|---|
| Neonatal EDF, 32 windows × 1024 samples × 21 channels | 1.718 ms | 0.663 ms | 61% lower latency |
| PhysioNet EDF, 32 × 1024 × 64 | 1.750 ms | 0.709 ms | 59% lower latency |
| WFDB-212, 32 × 8192 × 2 | 0.78–0.85 ms | about 0.24 ms | 68–72% lower latency |
| WFDB-212, 4 × 128 × 2 | about 0.03 ms | 0.065–0.075 ms | CPU wins |

EDF also favors CPU assembly at 4 × 128 for both recordings. Its 2 × 2 grid
separates batch sizes 4/32 and lengths 128/1024. Seven paths rotate through
seven positions over seven rounds, with three warmups and thirty calls per
path/round. WFDB uses six paths/six balanced rounds and two measurement series
with reversed workload order. Comparisons use the fastest measured non-GPU
assembly/decoding candidate, including compiled serial/four-thread WFDB controls.
Reported latencies are medians of round medians, not training speedups or
independent confidence intervals. Record reuse and overread affect EDF results;
logical payload counts are not measured physical disk traffic.

The useful result is an optional, source-aware GPU stage with a CPU alternative.
There is no universal CPU/GPU crossover rule. Select a representative consumer
with maintainers before integrating either path into the library.

## Supported semantics and limitations

- **EDF:** continuous ordinary EDF/EDF+C, known record counts, equal-rate voltage
  channels. Original int16 samples and calibration are preserved. Annotation
  words contribute to record strides even when interleaved with signal channels.
  Mixed rates, discontinuities, unsupported units/encodings and inconsistent
  layouts fail. Out-of-declared-range samples are preserved and reported.
- **WFDB-212:** one segment/file, two synchronized channels, one sample per
  channel/frame, common byte offset and zero skew. The GPU kernel fuses 12-bit
  unpacking, sign extension, missing-value handling and calibration. Digital
  -2048 remains the raw sentinel; physical outputs map it to NaN.
- Native EDF integers were compared exactly with edfio, and physical outputs
  with MNE. WFDB was checked digitally against the reference reader across
  all 1.3 million channel samples of MIT-BIH record 100. Physical comparisons
  use `rtol=2e-6, atol=1e-10 V`; missing masks must agree. Synthetic fixtures
  cover missing values, boundaries, duplicate requests and calibration metadata.
- Sources are CPU-read with warm OS caches, then transferred normally. There
  is no GPU-direct storage, hardware decoder, entropy decompression, cold-I/O
  measurement or demonstrated training benefit for these native paths. Results
  do not establish clinical-model accuracy or other GPUs/formats. Clocks were
  unlocked; repeated runs on one host are not independent datasets.

Primary specifications: [EDF](https://www.edfplus.info/specs/edf.html),
[EDF+](https://www.edfplus.info/specs/edfplus.html),
[WFDB encodings](https://physionet.org/physiotools/wag/signal-5.htm).

## Code responsibilities

| Responsibility | Modules in `profiling/` |
|---|---|
| EDF metadata and original integer access | `edf_source.py` |
| EDF record planning and CPU/GPU assembly | `edf_record_loading.py` |
| Independent MNE semantic checks | `edf_reference.py` |
| WFDB metadata/source access, compiled CPU and fused GPU processing | `wfdb212.py`, `wfdb212_cpu.py`, `wfdb212_cuda.py` |
| Affine numerical calibration | `signal_calibration.py` |
| Workload selection, correctness gates, measurements and reports | `benchmark_edf_records.py`, `benchmark_wfdb212.py` |
| Completed-work timing, hashing and transfer | `benchmark_support.py` |
| Read-only hardware/software provenance | `research_environment.py` |

Readers/decoders do not import benchmark orchestration. The EDF source reader
can be imported without MNE, PyTorch or Triton. Reference validation and numerical
processing are separate. There is no general loader framework or automatic dispatch.

## Evidence and provenance

- [results/edf.json](results/edf.json): all nine final EDF reports, including
  the eight configurations and summary.
- [results/wfdb212.json](results/wfdb212.json): all nine final WFDB reports,
  including both series, summary and full-record correctness checks.
- [results/measured-source.tar.gz](results/measured-source.tar.gz): source snapshot
  matching the historical report hashes, before helper extraction and this refactor.

Each JSON bundle's `reports` maps original relative filenames to unchanged
report objects; `original_file_sha256` records the original file-byte hashes.
No measurements were recalculated during consolidation. The historical code
was subsequently committed in `b852c549`; current names and responsibilities
have changed. Historical results describe that measured source, not timings of
this refactored tree. Extract the source snapshot separately to inspect its
modules. It is provenance, not maintained runtime code.

Superseded synthetic/cache/training experiments, traces and duplicated notes
are excluded from this contribution. They are not required by the native paths.
The cleanup preserved a verified backup outside the repository before removal.

## Reproduce and validate

See [REPRODUCING.md](REPRODUCING.md) for setup, checksums, tests and the two CLIs.
The recorded historical GPU validation is not a claim that CUDA was rerun on
this Mac. The retained tests check source semantics, calibration, record layout,
CPU/GPU outputs and architectural import boundaries. Fresh-install and CUDA
checks are required on an NVIDIA machine after this refactor.

Local cleanup validation (2026-11-15): **35 passed, 4 CUDA tests skipped**.
CPU EDF and WFDB reference/compiled-decoder tests actually executed. This used
the Mac's Python 3.10/PyTorch 2.14 environment, with isolated temporary medical
dependencies including MNE 1.10.2 (the recorded MNE 1.13.2 requires Python 3.11+),
edfio 0.4.18, pyEDFlib 0.1.42, WFDB 4.3.1 and Numba 0.62.1. It is a CPU
compatibility check, not reproduction of the original GPU measurements. Lint,
formatting, local imports, documentation links, evidence preservation and file
size/whitespace checks passed. No baseline library file was changed.
