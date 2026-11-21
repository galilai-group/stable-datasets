# Random EEG windows: storage before GPU tuning

Sami's review of [PR #162](https://github.com/galilai-group/stable-datasets/pull/162#pullrequestreview-6005673988)
prioritizes neonatal EDF, random windows for EEG classification, and a fair
comparison against efficient Lance layouts. The earlier native-format work
provides correctness checks and comparison paths; WFDB and mixed-rate EDF are
paused. Production builders and public features are unchanged.

## Current experiment

`python -m profiling.benchmark_signal_windows` has explicit `prepare`, `run`
and `plot` commands. See [REPRODUCING.md](REPRODUCING.md). Its paths are:

| Path | Read unit | Calibration |
|---|---|---|
| Current cache representation | Whole float32 recording through the production Arrow writer/backend/formatter | Prepared float32 volts |
| Native EDF memmap | Selected original data records assembled on CPU | Read-time CPU |
| Existing GPU path (CUDA runs only) | Original EDF records transferred, assembled on GPU | Read-time GPU |
| Lance recording blobs | Batched `read_blob_ranges`, no whole-blob Python slicing | Read-time CPU |
| Lance chunk rows | One deduplicated row take for the intersecting fixed-size chunks | Read-time CPU |

All CPU runs end at float32 NumPy batches in volts. CUDA runs end at float32
GPU tensors for **all five paths**, including transfers of CPU results. Never
compare those two endpoints in one throughput claim. Preparation and reference
checks are excluded from timing and reported separately. Each path occupies
every trial position once; completed-batch medians determine windows/second.
This is batch latency expressed as a rate, not overlapped loader throughput.

The current-cache baseline reproduces the neonatal **signal column's** nested
float32 recording-row representation, not the full builder, downloads or labels.
It groups requests to materialize each recording once per batch, avoiding a
weakened baseline that repeatedly reads the same recording. This checkout uses
Arrow by default; the benchmark records the baseline explicitly. It measures
current behavior rather than assuming a four-second request reads four seconds.

## Window contract and layouts

A request is `(recording index, start sample, length)`, selecting the half-open
interval `[start, start + length)`. It cannot cross a recording boundary. Reads
preserve request order, duplicates and source channel order. Different recording
shapes are supported by the storage API; the benchmark uses one EDF recording
at a time to produce a rectangular batch. Four-second windows use its actual
sample rate and must be sample-aligned.

Both immutable Lance layouts store time-major little-endian int16 samples and
recording metadata: sample count, rate, channel names, gain/offset in volts and
first dataset row. For blobs, the byte offset is `start * channels * 2` within
that recording's blob. For chunks, row offsets and integer division locate the
required rows, including the shortened last chunk. Repeated/overlapping chunk
requests share a row read. These experiment row indices assume no append or
compaction; there is no production cache migration or general signal feature.

The writer streams one blob/chunk at a time. The preparation command still
materializes one complete EDF recording on CPU; preparation memory/time must
not be confused with training read costs. Embedded EDF annotation metadata is
retained externally, but no seizure-label policy is invented. External neonatal
MAT labels, patient-separated training splits and model adaptation belong to
the later training experiment.

The recording-blob path requires genuine planned byte-range support, verified
with local Lance 13.0.0 and file format 2.2. No full-blob fallback is allowed.
API ranges and counters describe requested payloads; filesystem pages, Lance
coalescing and caches may cause different physical I/O. See the
[Lance blob API](https://lance.org/guide/blob/).

## Responsibility boundaries

- `signal_windows.py`: recording metadata, request validation and common window conversion.
- `lance_signal_windows.py`: immutable int16 storage and window selection, without training policy.
- `current_signal_cache.py`: production-format recording-row baseline.
- `benchmark_signal_windows.py`: preparation/measurement/plot orchestration and matched endpoints.
- `edf_source.py`, `edf_record_loading.py`: existing EDF source and CPU/GPU assembly.
- `signal_calibration.py`, `edf_reference.py`: numerical calibration and independent MNE validation.
- `benchmark_support.py`, `research_environment.py`: measurement utilities and provenance.

The existing GPU pinning/allocation behavior is intentionally unchanged, as
Sami requested. Historical WFDB modules remain dormant; no work is added there.

## Evidence and remaining work

Result JSON, figures, data and measured-source archives live **outside Git**.
Historical evidence was copied and checksum-verified outside the repository
before removal from this working contribution. Earlier timings do not establish
the performance of these new layouts. Source/code hashes accompany new reports.

Synthetic tests check real Lance reads, cross-chunk and recording boundaries,
duplicates, metadata/calibration, exact byte-range calls, deduplication payloads,
whole-recording baseline materialization and EDF-to-benchmark/plot execution.
Reference checks compare physical outputs with MNE before timing.

Only warm-source measurement is currently implemented. Reader reopening is not
called a cold-cache test. NFS, cold I/O, DataLoader worker settings and realistic
EEG-model training remain unmeasured. A user-supplied `nfs` label is not mount
verification. The next stage is to run the matched CUDA comparison on Hydra/NFS,
then establish model/forward-backward equivalence in stable-pretraining and
compare learning progress against wall time. No automatic CPU/GPU routing or
end-to-end training speedup is claimed.

Local validation of this stage: 31 tests passed, with two CUDA checks and two
paused-WFDB dependency checks skipped. Real Lance range/chunk reads and the
EDF-to-benchmark/plot smoke test executed. One unchanged neonatal `eeg4.edf`
was also checked against its published source digest and used for an exploratory
four-second-window CPU comparison (batch sizes 1/4/16/32, five repeats per path
in four balanced rounds). All paths passed MNE comparisons. Its report and
figure are external artifacts, not committed research results or evidence of
Hydra/NFS/GPU/training performance. The preparation source has 876,800 time
samples and 21 channels; one recording-row materialization therefore visits
18,412,800 scalar values, even for a short window. No baseline library file,
GPU assembly implementation or calibration algorithm was changed.
