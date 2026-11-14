# Persistent signal caches and a training comparison

The longer monitored follow-up protocol and profiler diagnostics are documented
in [SIGNAL_REPRODUCIBILITY.md](SIGNAL_REPRODUCIBILITY.md).

## Scope

This isolated prototype tests whether lower-transfer-volume integer signals
improve actual training throughput against an efficient float32 baseline. It
does not change library builders, define a production API, or implement GPU EDF
parsing. It currently measures one recording, not a collection-scale loader.

The numeric cache contract is independent of EDF: continuous equal-rate voltage
signals, explicit channel order, time-major `(time, channels)` layout and volts
after calibration. An individual source adapter must establish continuity,
sampling rates, channel semantics, units and calibration. Float32 storage is
already calibrated; int16 storage preserves native samples and carries affine
gain/offset vectors. These representations are alternatives, not a rule that
every physiological dataset should be quantized to int16.

## Implementation boundaries

- `signal_cache.py`: typed metadata, full-signal representation validation,
  exclusive persistence, checksums, read-only memory maps and ordered window
  copies into reusable buffers. No MNE or pyEDFlib dependency.
- `prepare_signal_cache.py`: EDF-specific preparation, original MNE reference,
  validation-report/source hash checks and explicit source-exception acceptance.
- `signal_training.py`: deterministic schedule, small Conv1D workload,
  reusable pinned/device buffers, CUDA stream/event synchronization, numerical
  training checks, complete-step measurements and JSON reporting.

Each cache directory contains `float32.npy`, `int16.npy` and `manifest.json`.
Both arrays are C-contiguous and have identical shape/channel order. The
manifest retains source validation and metadata, physical units, sample rate,
calibration and file hashes. Manifest publication happens after array writes.
Existing directories are never overwritten. Interrupted preparation may leave
an incomplete directory, which cannot be opened as a valid cache.

Writing and opening validate **every sample** with the existing combined
tolerance `abs(error) <= 1e-10 V + 2e-6 * abs(reference)`. Opening additionally
verifies file hashes, schema, dtype and memory order. This one-time verification
warms OS caches; none of the training measurements represents cold disk I/O.

## Source preparation

The neonatal source and compatibility findings are documented in
[NEONATAL_EDF_VALIDATION.md](NEONATAL_EDF_VALIDATION.md). Preparation requires
the previous successful validation report and exact source checksums. The
original EDF remains MNE's reference. The integer path uses the already
identified derivative, with unchanged signal bytes and explicit acceptance of
the header inconsistency and out-of-declared-bounds samples. No source repair
is performed by cache preparation. Standard source reading remains strict.

This paired-cache prototype includes source-specific annotations in provenance;
it does not introduce a general annotation storage format. The training workload
does not use clinical labels or claim seizure-detection accuracy.

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.prepare_signal_cache /tmp/stable-datasets-neonatal/eeg4.edf --digital-source /tmp/stable-datasets-neonatal/eeg4-validation-copy.edf --validation-report signal-loading-neonatal-validation.json --accept-source-exceptions --output-dir /tmp/stable-datasets-neonatal/paired-cache-v2
```

Use a new directory if that path exists. Temporary source data and dependencies
can be recreated using the preceding EDF documentation. On this machine,
preparation including source hashing, full reading/validation and cache writing
took 1.391 s for 876,800 samples x 21 channels. Array file sizes:

| Representation | File bytes |
|---|---:|
| Float32 | 73,651,328 |
| Int16 | 36,825,728 |

This halves array storage, without accounting for the small shared manifest.
The paired experiment stores both files; a production deployment would choose
a representation rather than necessarily duplicating data.

## Training workload and measurement contract

`SignalRegressor` has two Conv1D layers (64 hidden channels), GELU activations,
global average pooling and a channel-wise linear output. It predicts each
window's per-channel mean in mV using MSE and SGD at learning rate 0.001.
Forward, target construction, backward and optimizer updates are all timed.
This provides actual GPU compute, but is not a representative validated seizure
model. All 21 original channels, including auxiliary ECG/respiration, are retained.

Every path uses identical scheduled windows, seed, initial model weights,
optimizer settings and warmup prefix. CuDNN deterministic mode is enabled;
CuDNN benchmarking and TF32 are disabled. Before timing, inputs, first-batch
loss and updated parameters are compared between float32/int16 CUDA paths.
Regression tests additionally compare final losses after multiple training
steps across all five paths, exercising buffer reuse.

Paths:

1. `resident`: the complete float32 schedule is already on GPU, isolating model
   work plus Python launch overhead. This is a reference, not a scalable loader
   or a theoretical throughput ceiling.
2. `float32_serial` and `int16_serial`: gather into reusable pinned buffers,
   transfer, optionally calibrate on GPU and train, synchronizing every step.
3. `float32_overlap` and `int16_overlap`: same work with two buffers and a
   dedicated transfer stream. Events protect host reuse and device-buffer reuse;
   the compute stream waits for each batch. This permits transfer/model overlap,
   but does not prove that every transfer overlaps or hide all CPU gathering.

Only the entire timed sequence is synchronized for overlapping paths. Setup,
cache validation, allocation of persistent buffers/model/optimizer and initial
GPU calibration parameters are excluded. Allocations occurring during training
and int16 calibration are included. Loss logging occurs after synchronization.
No background DataLoader workers are used. All GPU-resident schedule tensors
remain allocated during every path so device-memory pressure is comparable;
memory use grows with schedule length and is not a production strategy.

Trial order rotates, so with five repeats each path appears in every position.
JSON retains all trial timings, final losses and peak PyTorch allocated bytes,
as well as configuration, schedule/cache/code hashes and library versions.
It does not measure NVML utilization, CPU wait breakdown, power, cold I/O or
total driver/device memory. A profiler trace is needed to establish the actual
overlap and identify remaining bottlenecks.

## Run and validate

Use a normal terminal with GPU access; the agent sandbox hides this GPU.

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py stable_datasets/tests/test_neonatal_edf_validation.py stable_datasets/tests/test_signal_cache.py
python -m profiling.signal_training /tmp/stable-datasets-neonatal/paired-cache-v2 --steps 300 --repeats 5 --warmup 5 --cpu-threads 1 --output signal-training-neonatal-final-t1.json
python -m profiling.signal_training /tmp/stable-datasets-neonatal/paired-cache-v2 --steps 300 --repeats 5 --warmup 5 --cpu-threads 4 --output signal-training-neonatal-final-t4.json
```

Choose new output paths on repetition. Default batch dimensions are
32 x 1024 x 21; default CLI run lengths are smaller for quick checks.

## Exploratory runs

All earlier outputs were preserved. `signal-training-neonatal.json` used the
initial v1 cache, whose MNE float32 array was Fortran-contiguous while int16 was
C-contiguous. This confounds the representation comparison; do not use that
run to claim an int16 advantage. Cache v2 normalizes and enforces C order for
both arrays. A regression fixture deliberately supplies Fortran-order float32
input to verify this behavior.

`signal-training-neonatal-contiguous.json` used v2 but only 50 steps per trial.
`signal-training-neonatal-t1.json` used 300 steps before extraction of reporting
helpers. The final reports named above identify the final source by checksum.

## Final observations, 2026-11-14

The final implementation passed **40 tests in 5.33 s**, including all CUDA
checks, with no skips. Ruff lint/format checks passed. The source exceptions
remain explicitly recorded in the cache; they have not become library defaults.

Each final run uses 300 steps per trial, five repeats per path and five warmup
updates, with identical 32 x 1024 x 21 input windows on the RTX 4060 Laptop GPU.
Environment and neonatal source identity match the preceding experiments.
Base Git revision remains `65bf7dcb5cbf214d347c78dee3639bf3e2fb8047`; all research
changes are uncommitted. Source checksums in both reports match the final code.

| Path | 1 CPU thread, median steps/s | 4 CPU threads, median steps/s |
|---|---:|---:|
| GPU resident reference | 1025.2 | 1011.0 |
| Float32 serial | 492.9 | 527.9 |
| Int16 serial | 555.2 | 584.2 |
| Float32 overlap | 675.7 | 705.8 |
| Int16 overlap | 775.6 | 818.6 |

Compared with the stronger overlapped float32 baseline, int16 was approximately
14.8% faster at one CPU thread and 16.0% faster at four in these final runs.
The final loss was `0.005431880708783865` for all paths and trials. This is a
systems correctness result, not evidence of clinical learning or accuracy.

Run variability is material: at four threads, individual float32-overlap trials
ranged from 620.8 to 733.5 steps/s and int16-overlap from 636.2 to 874.9 steps/s.
The earlier 300-step `signal-training-neonatal-t1.json` run even favored
float32-overlap (571.8 versus 536.5 steps/s). Changing thermal/power/system state
was not controlled or profiled; its contribution versus code/runtime overhead
has not been isolated. The final medians therefore show a promising case, not a
robust universal performance conclusion or a confidence interval.

Peak allocated GPU memory is approximately 809 MiB for the resident path and
814–815 MiB for cache paths, including the common resident schedule kept alive
for all paths. This is not an isolated measurement of loader memory usage.

## Next research step

Interpret int16 against the **best float32 training path**, not only its serial
baseline. Repeat on more subjects and at least one further source with known
metadata, then vary model compute, window/batch sizes and data volume. Profile
actual overlap before selecting a default. Demonstrate source compatibility
without silent fixes and include preparation cost. An integer cache can still
save storage even if it does not improve training speed.
