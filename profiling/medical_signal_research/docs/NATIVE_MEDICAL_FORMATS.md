# Native medical formats: source-to-GPU investigation

The completed cross-format feasibility assessment is in
[RESEARCH_CONCLUSION.md](RESEARCH_CONCLUSION.md); packed WFDB-212 results are in
[WFDB212_RESULTS.md](WFDB212_RESULTS.md).

The follow-up record-batched CPU/GPU experiment, including optimized CPU
baselines and small-batch counterexamples, is in
[EDF_RECORD_LOADING.md](EDF_RECORD_LOADING.md).

## Research objective

Investigate Sami's original question: how can physiological signals be loaded
from their medical source formats onto the GPU quickly, and which interpretation
or decoding steps could usefully run there? Previous processed-cache results
remain a separate intermediate finding. They do not establish native GPU decoding.

The checkpoint for this phase is commit
`cd5e3b259baede94fd659696b7d438ee35dfd524`. This experiment adds only an isolated
reader adapter/benchmark, regression tests and documentation. Existing builders
and cache defaults are unchanged. No Git operations were performed by the agent.

## Format properties and candidate operations

| Source format | Source representation | Operations to investigate |
|---|---|---|
| Ordinary EDF / EDF+C | Uncompressed signed 16-bit samples in per-channel blocks within data records | Record selection, deinterleaving/layout conversion, transfer, affine calibration |
| WFDB format 16 | Little-endian signed 16-bit samples, potentially multiplexed | Header interpretation, channel/layout selection, transfer and calibration |
| WFDB format 212 | Two signed 12-bit values packed into three bytes | Bit unpacking, sign extension, layout, transfer and calibration |
| WFDB formats 508/516/524 | FLAC-compressed signals | Actual codec decompression as well as signal layout/calibration; GPU feasibility remains uninvestigated |

Primary specifications: [EDF](https://www.edfplus.info/specs/edf.html),
[EDF+](https://www.edfplus.info/specs/edfplus.html),
[WFDB signal encodings](https://wfdb.io/spec/signal-files.html),
[WFDB headers](https://wfdb.io/spec/header-files.html).

EDF therefore does not have an entropy-decoding workload equivalent to video
decoding. Its potentially useful GPU work is selection/layout/calibration.
WFDB packing/compression supplies distinct candidate workloads, not evidence
that a GPU implementation would outperform CPU implementations.

## Native EDF adapter and source compatibility

`profiling/native_edf_loading.py` uses existing source readers:

- MNE `read_raw_edf(preload=False)` for the physical-value reference and an
  independent CPU-calibration baseline.
- edfio 0.4.18 `read_edf(lazy_load_data=True)` and the public
  `EdfSignal.get_digital_slice()` API for native integer windows from source EDF.

Documentation: [edfio reading](https://edfio.readthedocs.io/en/latest/generated/edfio.read_edf.html),
[signal slice API](https://edfio.readthedocs.io/en/latest/generated/edfio.EdfSignal.html).
Inspection of the installed reader confirms that lazy slices copy selected
record/channel regions from a memory-mapped EDF source. The adapter never
accesses full-signal `.digital`/`.data` properties and does not use processed NPY
or Arrow caches. Reading remains on CPU; CUDA performs calibration only.

An important improvement over the previous experiment: edfio reads both
unmodified original files, including neonatal `eeg4.edf`, whose EDF+C marker
without an annotation channel caused pyEDFlib to reject it. No normalized
derivative is used. This shows reader compatibility, not source conformance.
Values outside declared digital bounds are preserved, counted in selected
windows and compared against MNE rather than clipped or reinterpreted.

The adapter accepts continuous, equal-rate voltage signals only. It rejects
other encodings, mixed rates, unsupported units and discontinuities. It checks
channel order, sample rate/count, recording start and embedded EDF annotations.
External neonatal MAT labels are not reprocessed here; their prior validation
is documented in [neonatal source validation](../archive/docs/NEONATAL_EDF_VALIDATION.md).

## Measurement boundaries

Every batch path includes original-source window access, record/channel copying,
CPU layout conversion, allocation and any pinning/transfer/calibration named in
that path. No converted recording is reused between calls. The source windows
are read again on every invocation; the operating system page cache is warm.
No physical cold-disk throughput, network I/O, training or overlapped pipeline
throughput is measured.

Two scenarios are kept separate:

1. Reused readers: parsing/setup and GPU calibration parameter placement occur
   once outside timing; source-window access remains inside each call.
2. Reopen per batch: reader opening/header interpretation are included. The
   integer GPU variant also creates and transfers calibration parameters anew.

CPU header-only diagnostics are separate. The edfio open diagnostic times object
creation/mapping, excluding result destruction; the MNE diagnostic includes
close. Do not treat their ratio as a matched whole-reader lifecycle comparison.

All complete GPU operations are compared against the same MNE float64 windows
before timing, with `rtol=2e-6`, `atol=1e-10 V`. Timings use three warmups and
30 repeats, with synchronized completed-batch latency and one PyTorch CPU thread.
Stages run in fixed order; these are diagnostic measurements, not a balanced
reproduction study or a statistical speedup claim. Independent stage medians
must not be subtracted as an exact cost decomposition.

## Reproduce

The existing isolated dependencies remain under `/tmp/stable-datasets-edf-deps`.
Install the additional reader there, preserving the active environment:

```bash
python -m pip install --target /tmp/stable-datasets-edf-deps --no-deps edfio==0.4.18
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.native_edf_loading /tmp/stable-datasets-neonatal/eeg4.edf --device cuda --repeats 30 --output native-edf-neonatal.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.native_edf_loading /tmp/stable-datasets-edf-data/S001R01.edf --device cuda --repeats 30 --output native-edf-physionet.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py stable_datasets/tests/test_neonatal_edf_validation.py stable_datasets/tests/test_signal_cache.py stable_datasets/tests/test_signal_experiment.py stable_datasets/tests/test_native_edf_loading.py
```

Choose new output names on repetition. Source-download commands and checksums
are in the earlier EDF documents. No further dataset download was needed.
CUDA commands require GPU access outside the agent sandbox.

## Observations, 2026-11-14

Hardware is the same RTX 4060 Laptop GPU. Python 3.13.9, PyTorch 2.15.0+cu132,
MNE 1.13.2, NumPy 2.3.5 and edfio 0.4.18. JSON reports record source and code
SHA256 digests, versions and deterministic window starts. Original source
checksums were unchanged after the complete experiment.

Both batches contain 32 windows of 1,024 samples, including recording edges and
an EDF data-record boundary. Neonatal EEG has 21 channels at 256 Hz; PhysioNet
has 64 channels at 160 Hz. These represent different payload sizes/durations,
not directly comparable workloads across datasets.

| Reused-reader stage, median ms | Neonatal | PhysioNet |
|---|---:|---:|
| MNE source windows -> CPU float32 | 6.698 | 15.623 |
| edfio source windows -> CPU int16 | 4.011 | 13.660 |
| edfio source windows -> CPU-calibrated float32 | 4.651 | 15.446 |
| MNE source windows -> GPU float32 | 7.343 | 17.523 |
| edfio source -> CPU calibration -> GPU | 5.144 | 18.868 |
| edfio source -> GPU calibration | 4.458 | 14.962 |

The fastest measured CPU-calibration source-to-GPU path is edfio for neonatal
EEG and MNE for PhysioNet. Against that stronger baseline per recording, GPU
calibration reduces median completed-batch latency by 13.3% and 14.6%
respectively in these diagnostic runs. These are latency reductions, not the
earlier training-throughput percentages or native GPU decoding results.

| Reopen-per-batch path, median ms | Neonatal | PhysioNet |
|---|---:|---:|
| MNE source -> GPU float32 | 9.324 | 21.388 |
| edfio source -> CPU calibration -> GPU | 5.685 | 18.291 |
| edfio source -> GPU calibration | 4.927 | 15.731 |

CPU open/map diagnostics were 0.187/0.342 ms for edfio and 1.239/2.888 ms for
MNE open/close. Source integer-window access alone is close to the full
source-to-GPU latency, indicating substantial source/host work rather than an
arithmetic-heavy EDF decoder. This does not isolate physical storage cost.

Maximum selected CPU error against MNE float64 was approximately `6.21e-10 V`
for neonatal and `2.96e-11 V` for PhysioNet. All GPU paths passed the same
tolerance. The neonatal batch contained 127 out-of-declared-range sample
occurrences; these are selected-window occurrences, not a new full-record count.

**49 tests passed in 5.48 s**, with no skipped CUDA tests. Four new tests cover
request order/duplicates, record boundaries, calibration/annotation agreement,
unsupported rates/encodings, unchanged inconsistent sources and every complete
GPU path. Ruff lint/format checks passed. A three-repeat CPU smoke run passed
before the full measurements.

## Bottleneck evidence and next experiment

A separate instrumented CPU profile of ten neonatal native-window batches
recorded 6,720 `get_digital_slice` calls: 32 windows x 21 channels x 10 batches.
It also showed repeated header-field decoding and memory-map slicing/flattening.
Profiler timings are not source-latency estimates and were not used as speedup
measurements. Call counts identify a concrete candidate for improvement.

The next EDF experiment should compare record-aware **batched source access**
with a strong vectorized CPU baseline. Only then compare transferring original
record blocks to GPU for layout/window extraction/calibration. Preserve source
bytes, validate offsets across record boundaries, and measure read amplification
from transferring complete records. Existing edfio metadata can remain the
header/reference layer; there is no justification yet to replace it with a
general custom parser.

In parallel with that format-specific decision, the next distinct format worth
testing is a small WFDB-212 recording, because packed samples introduce actual
bit-unpacking work. Its CPU/GPU feasibility remains a hypothesis. FLAC-based
WFDB is a separate codec investigation, not an automatic extension of EDF work.
