# Native EDF record batching: CPU versus GPU assembly

The follow-up 2x2 study separating batch size and window length is documented in
[EDF_WORKLOAD_LIMITS.md](EDF_WORKLOAD_LIMITS.md).

## Relation to Sami's task

This experiment investigates the original medical source format directly:
unchanged EDF file -> selected original records -> GPU signal windows in volts.
It follows the source-access bottleneck found in `native_edf_loading.py` and
does not use a processed cache or the earlier normalized EDF derivative.

EDF stores signed 16-bit channel blocks inside each data record. The experiment
tests whether record selection, window/layout assembly and calibration benefit
from GPU execution. This is not entropy decoding, a hardware decoder, direct
GPU storage access or a production loader. CPU memory-mapped reads and H2D
transfers remain part of every applicable path.

Primary layout references: [EDF](https://www.edfplus.info/specs/edf.html),
[EDF+](https://www.edfplus.info/specs/edfplus.html).

## Implementation and validation boundary

`edf_record_loading.py` keeps metadata interpretation in the existing edfio
reader. Its narrow payload-layout adapter reads the fixed header, labels and
sample counts to derive record size and channel offsets. These are cross-checked
against edfio's public metadata and the exact source-file length. It accepts
known positive record counts and continuous equal-rate voltage signals only;
unknown counts, unsupported encodings, mixed rates, discontinuities, trailing
bytes and truncation are rejected rather than guessed.

Annotation-channel words contribute to each record's stride, including when an
annotation channel lies between ordinary channels. They are excluded from the
numerical output. Metadata/annotations remain CPU-interpreted and are checked
against MNE. Native samples outside declared calibration bounds are preserved
as in the preceding source experiment. Compatibility does not establish source
conformance or medically validate the recording.

The code separates:

- `RecordLayout`: verified payload offsets/strides in int16 words.
- `WindowPlan`: requested starts and the union of necessary record IDs.
- `BatchedEdfRecords`: read-only source mapping and CPU access strategies.
- `GpuRecordAssembler`: transfer original records, select windows/layout on CUDA
  and apply the unchanged channel calibration parameters.
- `benchmark_edf_records.py`: correctness checks, operation definitions, balanced
  latency measurement and report serialization.

CPU strategies include direct broadcasted sample/channel indexing and a stronger
record-oriented baseline. The latter reads each selected record once, extracts
channel blocks, creates a contiguous time-major record array and copies each
requested window as a contiguous slice. Because every record within a requested
window belongs to the sorted union, gaps between different windows can be
omitted without breaking any individual window's continuity.

The GPU path uses the same property: only the starting record of each window
needs a search. It builds sample/channel offsets on the GPU and gathers from
transferred original record bytes. It preserves duplicate requests and original
request order. All calibration parameters and constant channel offsets are
placed on the GPU once before timing.

## Measurement contract

Seven paths are compared: existing-library CPU/GPU calibration, direct NumPy
window assembly with CPU/GPU calibration, optimized CPU record assembly with
CPU/GPU calibration, and GPU record assembly/calibration.

Every complete batch measurement includes plan construction, source access,
layout/index construction, allocation, per-batch pinning, transfers and the
calibration specified by the path. GPU paths also transfer their dynamic starts
and record IDs. Source-reader setup is reused and excluded. Synchronization
measures completed batch latency; there is no model, prefetching or overlapped
training pipeline here. Source hashing/validation warms the OS page cache, so
these are not cold-disk or NVMe bandwidth measurements.

The seven paths start in a fixed seeded permutation and rotate across seven
rounds, balancing each path across all positions. Each round has three warmups
and thirty completed-batch measurements per path. The reported statistic is
the median of seven round medians. This is a diagnostic, not an independent
multi-session performance study or a training-speedup result.

Every CPU layout is compared bit-for-bit with edfio's native integer windows.
Every complete GPU path is compared with independent MNE float64 windows using
`rtol=2e-6, atol=1e-10 V`. Source hashes are checked before and after measuring.

## Reproduce

Use the existing isolated MNE/edfio/pyEDFlib environment documented in
`NATIVE_MEDICAL_FORMATS.md`. No further downloads were needed.

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-neonatal/eeg4.edf --output edf-records-neonatal-final.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-edf-data/S001R01.edf --output edf-records-physionet-final.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-neonatal/eeg4.edf --batch-size 4 --length 128 --output edf-records-neonatal-small.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-edf-data/S001R01.edf --batch-size 4 --length 128 --output edf-records-physionet-small.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py stable_datasets/tests/test_neonatal_edf_validation.py stable_datasets/tests/test_signal_cache.py stable_datasets/tests/test_signal_experiment.py stable_datasets/tests/test_native_edf_loading.py stable_datasets/tests/test_edf_record_loading.py
```

Run CUDA outside the restricted agent sandbox. Use fresh output filenames on
repetition. All earlier results are preserved. The files without `-final` or
`-small` in their names are exploratory runs with a weaker CPU layout strategy;
they are not used for the final comparisons below. Small initial smoke outputs
were temporary checks. Earlier `edf-records-*.json` reports now live in
`profiling/medical_signal_research/archive/results/`.

## Results, 2026-11-14

Same original files and RTX 4060 Laptop GPU as before; Python 3.13.9,
PyTorch 2.15.0+cu132, NumPy 2.3.5, MNE 1.13.2, edfio 0.4.18. One PyTorch CPU
thread. The base checkpoint is `cd5e3b259baede94fd659696b7d438ee35dfd524`;
all new prototype changes are uncommitted. Reports retain source/code hashes,
window starts, layouts, payload sizes, errors and all round statistics.

Large batch, 32 x 1024 values per channel, median ms:

| Complete source-to-GPU path | Neonatal, 21 channels | PhysioNet, 64 channels |
|---|---:|---:|
| Library window access, CPU calibration | 5.281 | 17.487 |
| Library window access, GPU calibration | 4.479 | 14.807 |
| Direct CPU gather, CPU calibration | 3.785 | 11.695 |
| Direct CPU gather, GPU calibration | 3.018 | 9.175 |
| Optimized CPU record assembly, CPU calibration | 2.310 | 3.898 |
| Optimized CPU record assembly, GPU calibration | 1.531 | 1.681 |
| GPU record assembly and calibration | 0.566 | 0.617 |

The strongest measured non-GPU-assembly path already uses GPU calibration.
Against that stronger hybrid baseline, moving layout/window assembly to GPU
reduces median batch latency by about 63% in both large-batch cases (latency
ratios approximately 2.71 and 2.72). This does not imply a 2.7x training speedup.

Small batch, 4 x 128 values per channel:

| Complete source-to-GPU path | Neonatal median ms | PhysioNet median ms |
|---|---:|---:|
| Best measured path: direct CPU gather + CPU calibration | 0.111 | 0.197 |
| GPU record assembly and calibration | 0.209 | 0.215 |

GPU assembly loses at these measured small-batch points. Its neonatal latency
is approximately 89% higher; the PhysioNet difference is smaller, about 9%, and
needs cautious interpretation given run variability. No universal dispatch
threshold is inferred from only two batch shapes.

### Payload reuse and overread

GPU record transfer includes each required complete record once, including
annotation bytes, rather than transferring duplicated output windows.

| Workload | Original record bytes | Logical requested int16 window bytes | Record bytes / unique requested signal bytes |
|---|---:|---:|---:|
| Neonatal large | 1,559,040 (145 records) | 1,376,256 | 1.220 |
| PhysioNet large | 1,259,040 (61 records) | 4,194,304 | 1.008 |
| Neonatal small | 53,760 (5 records) | 21,504 | 2.500 |
| PhysioNet small | 103,200 (5 records) | 65,536 | 1.575 |

PhysioNet is only 61 seconds long: the large batch's overlapping windows reuse
nearly the entire recording. Deduplicated record transfer is therefore much
smaller than its logical window payload. The neonatal large batch has less
reuse and reads about 22% more bytes than unique requested signal values.
These quantities are logical source/transfer payloads, not measured physical
storage traffic. Small-window overread and fixed GPU launch/transfer overhead
are plausible contributors to the small-batch reversal, not an isolated causal
decomposition.

### Correctness and limitations

**58 tests passed in 5.32 s**, with no skipped CUDA tests. Nine new tests cover
exact raw samples/order/duplicates, record boundaries, annotation channels both
last and interleaved, compact record unions with gaps, full-record windows,
invalid requests, truncated files/unknown counts and every complete GPU path.
Ruff lint/format checks passed. Both original source files remained unchanged.
All large-batch GPU paths had the same maximum MNE error: about `6.21e-10 V`
for neonatal and `2.96e-11 V` for PhysioNet, within the combined tolerance.

Only two recordings, two batch shapes and warm source reads have been tested.
The experiment does not support mixed rates, discontinuous recordings, arbitrary
medical formats or direct GPU storage reads. The prototype uses multiple
PyTorch CUDA operations rather than a fused decoder kernel. Persistent pinned
buffers, cold storage, model contention and end-to-end training are unmeasured
for this record-level path.

## Research conclusion and next decision

This is evidence directly aligned with Sami's native-format question: GPU
window/layout assembly from original medical record blocks can help at a useful
batch size, while an efficient CPU path wins for small requests. The useful
operation is source-format-aware selection/layout/calibration, not EDF entropy
decompression.

Retain both CPU and GPU candidates. Next determine the crossover across request
size, window overlap and channel count, using additional recordings and a
representative consumer. Separately, a packed format such as WFDB-212 would test
actual bit-unpacking rather than assuming that EDF's result transfers to every
medical format. Do not integrate a GPU-only default or claim general medical
decoding acceleration from these results.
