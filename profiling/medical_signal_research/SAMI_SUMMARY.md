# Native medical signals on GPU — summary for Sami

## Objective and outcome

We investigated fast loading and GPU interpretation of native physiological
signal formats, following your suggestion of an “NVDEC equivalent” for EEG or
ECG. The feasibility study covers two operations:

- **EDF/EDF+C:** selecting original data records, assembling signal windows and
  applying voltage calibration on the GPU.
- **WFDB-212:** unpacking signed 12-bit samples from three-byte pairs, handling
  missing values and applying voltage calibration in a fused GPU kernel.

Both work correctly for the supported subsets. GPU execution reduces loading
latency for larger tested batches, while efficient CPU implementations win
small requests. This supports an optional GPU stage with CPU fallback.

## Measured results

Measurements end at usable float32 GPU tensors and include source-window
gathering, allocation and transfer. Header parsing and constant metadata setup
are outside timing. The comparisons use efficient CPU alternatives, including
compiled serial and four-thread WFDB decoders.

| Workload | Best CPU/hybrid path | GPU path | Result |
|---|---:|---:|---|
| EDF, neonatal EEG, 32 windows × 1024 samples/channel | 1.718 ms | 0.663 ms | 61% lower latency |
| EDF, PhysioNet EEG, 32 × 1024 | 1.750 ms | 0.709 ms | 59% lower latency |
| WFDB-212, MIT-BIH ECG, 32 × 8192 × 2 channels | 0.78–0.85 ms | about 0.24 ms | 68–72% lower latency across two runs |
| WFDB-212, 4 × 128 × 2 channels | about 0.03 ms | 0.065–0.075 ms | CPU wins |

CPU assembly also wins both tested EDF sources at 4 × 128. EDF uses seven
balanced-order rounds per configuration; WFDB uses six per run. These are
completed-batch latency results, not demonstrated training speedups.

## Correctness and scope

Native integer EDF windows match edfio exactly; physical GPU outputs match MNE
within the documented numerical tolerance. WFDB decoding matches the reference
reader bit-for-bit across the full record's 1.3 million channel samples;
physical values and missing-value semantics also pass reference checks.
After cleanup, **74 regression tests passed with no skips, including CUDA**,
and twelve executable checks passed, including short GPU training and full-record
WFDB decoding. Original recordings and historical measurements are preserved.

The study uses two EEG recordings and one ECG recording, warm OS caches and one
RTX 4060 Laptop GPU. Metadata and file access remain on CPU, followed by normal
host-to-device transfer. EDF work is selection/layout/calibration of uncompressed
samples; WFDB-212 adds packed-bit decoding. Hardware decoding, entropy
decompression, direct GPU storage access, cold-I/O gains and generalization to
other devices or arbitrary medical formats have not been established.

The implementations remain standalone research prototypes; the original
production library files are unchanged. Earlier processed-cache experiments
are archived separately from these native-format findings.

## Recommended next decision

The scoped feasibility investigation is complete. Before integrating a loader,
we should agree on the target format subset and actual consuming workload,
then measure its end-to-end benefit and CPU/GPU crossover. The current evidence
supports retaining both paths; it does not establish a universal routing rule.

Details and reproduction: [research conclusion](docs/RESEARCH_CONCLUSION.md),
[EDF workload study](docs/EDF_WORKLOAD_LIMITS.md),
[WFDB results](docs/WFDB212_RESULTS.md), and [validation](VALIDATION.md).
