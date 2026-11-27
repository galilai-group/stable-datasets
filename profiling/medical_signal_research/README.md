# Random EEG windows: storage before GPU tuning

This source-checkout experiment compares random neonatal EDF windows with
production-format Arrow and two immutable Lance layouts. Production builders
and public features are unchanged. See [REPRODUCING.md](REPRODUCING.md) for
installation and validation; `profiling` is not installed as a package.

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
retained externally. The separate CNN diagnostic below owns its neonatal MAT
annotation policy; patient-separated evaluation is outside this experiment.

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
- `neonatal_targets.py`: expert annotations and the diagnostic's explicit window-label policy.
- `training_windows.py`: batched CPU readers that open independently in spawned workers.
- `benchmark_signal_training.py`: small CNN, matched training runs and data-wait measurements.
- `edf_source.py`, `edf_record_loading.py`: existing EDF source and CPU/GPU assembly.
- `signal_calibration.py`, `edf_reference.py`: numerical calibration and independent MNE validation.
- `benchmark_support.py`, `research_environment.py`: measurement utilities and provenance.

The existing GPU pinning/allocation behavior is unchanged. The paused WFDB
experiment and superseded seven-path EDF ablation CLI have been removed; native
EDF correctness tests import the owning reader, calibration and MNE modules.

## Small CNN and data loading diagnostic

`python -m profiling.benchmark_signal_training` consumes the existing prepared
stores and the published neonatal MAT annotations. It uses the same seeded
window schedule, targets, model initialization and AdamW settings for every
loader. CPU paths support different DataLoader worker counts. Workers use
`spawn`, open their own readers and return CPU tensors; CUDA remains in the
training process. The unchanged GPU assembler is compared with zero workers.
One dataset item is a full batch, preserving the recording-row baseline's
once-per-recording materialization behavior.

The diagnostic uses whole-second windows because the three experts' annotation
time base is one second. Its explicit label policy is a majority vote per second
and a positive window if any included second is positive. This is an experiment
choice, not a clinical evaluation standard. The small CNN selects channels named
`EEG`, excludes auxiliary signals and applies the same fixed scaling to every
path. It is a single-recording systems diagnostic; patient-generalization
evaluation is outside this diagnostic. EEG-JEPA model adaptation and training
live in the sibling stable-pretraining example.

The first batch is checked against MNE. Every measured training trajectory must
match the reference loss trajectory within reported tolerances. Two model
warmups precede measurement; model weights and optimizer state reset per trial.
Training wall time includes worker startup, data delivery and synchronized
forward/backward/optimizer steps. Data-ready time includes host-to-device work,
while loader-wait time isolates the `next()` call. Workers can prefetch during
compute, but synchronization at step boundaries means this is not a fully
asynchronous training pipeline. Reader shutdown is outside the measured interval.

The CLI saves a flushed progress journal beside the final report. Validated
trials are retained between measurements, even if a later reader fails. Failure
records include the batch number, phase and partial observations; partial or
unvalidated trials must not be pooled with successful trials. `--paths` can
isolate a reader for diagnosis without repeating the full comparison. Worker
timeouts remain 60 seconds; a timeout alone does not identify its underlying cause.

CNN batches use different windows across steps. First-batch validation and
replaying the schedule affect caching; the diagnostic does not verify residency
of every requested page. Keep this condition distinct from the window-only
benchmark, which warms the same request before timing its repetitions.

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
called a cold-cache test. A user-supplied `nfs` label is not mount verification;
retain independent mount evidence beside each report. Inspect worker startup
and per-step data-ready/compute times before interpreting CNN training wall
time. The sibling `stable-pretraining/examples/eeg_jepa` example retains reference
forward/backward equivalence, contextual preprocessing and a matched EDF pilot.
Old-cache CPU versus new-GPU EEG training, further Lance integration, GPU
preprocessing and buffer tuning remain future work. No automatic CPU/GPU
routing or end-to-end training speedup is claimed.

Historical exploratory CPU window measurements used the published `eeg4.edf`
digest and batch sizes 1/4/16/32, five repeats and four balanced rounds. Their
report and figure remain external artifacts; they are not measurements of this
cleanup, Hydra/NFS/GPU evidence or training results. The source has 876,800 time
samples and 21 channels, so one recording-row materialization visits 18,412,800
scalar values even for a short window. New reports retain their own code hashes.
