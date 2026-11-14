# Research conclusion: native physiological formats on GPU

## Scope completed

Sami asked to investigate time-series/medical file formats and fast loading or
direct GPU decoding, described as an “NVDEC equivalent” for physiological data.
The completed feasibility study answers this for two concrete source-format
operations, with existing readers as semantic references and strong CPU controls:

1. EDF/EDF+C record selection, window/layout assembly and affine calibration.
2. WFDB-212 packed 12-bit decoding, sign extension, missing-value handling and
   voltage calibration.

This completes the scoped exploratory investigation, not a production universal
medical decoder. It does not establish that a dedicated hardware block exists
or that all medical formats should use GPU decoding. Metadata remains CPU-read;
source bytes pass through CPU memory before ordinary H2D transfer.

## Evidence and decision

| Investigated operation | Supported result | Decision |
|---|---|---|
| Native EDF record/window assembly | GPU helps batched requests; efficient CPU wins small requests; equal output byte counts can prefer different paths | Preserve both source-aware CPU and GPU candidates |
| Native WFDB-212 bit decoding | Fused GPU kernel beats fused four-thread CPU for medium/large tested batches; fused serial CPU wins small requests | A GPU packed-decoder is useful selectively, not as a blanket replacement |
| Earlier processed-cache work | Demonstrated conversion/transfer opportunities but bypassed source-format work | Keep as a separate intermediate finding, not the headline answer to Sami |

Native EDF's 2x2 study used unchanged neonatal and PhysioNet EEG recordings.
At 32x1024, GPU assembly reduced latency about 59–61% versus the strongest
measured CPU-assembly/hybrid alternative. At 4x128, CPU assembly won. This is
layout/selection work on uncompressed source blocks, not entropy decompression.
See [EDF_WORKLOAD_LIMITS.md](EDF_WORKLOAD_LIMITS.md) and
[EDF_RECORD_LOADING.md](EDF_RECORD_LOADING.md).

Native WFDB-212 used original MIT-BIH ECG record 100, independently checked
against WFDB. At 32x8192x2 the GPU completed source-to-GPU decoding in about
0.24 ms versus 0.78–0.85 ms for the strongest compiled CPU/transfer path, a
68–72% latency reduction across two sequential runs. At 4x128x2 the serial CPU
path took about 0.03 ms versus 0.065–0.075 ms for GPU decoding. The entire
1.3-million-sample record was verified digitally bit-for-bit; physical values,
channel order, units and invalid-sample semantics were validated. See
[WFDB212_RESULTS.md](WFDB212_RESULTS.md).

All reported native measurements include source-window gathering and transfer
costs. Prepared-input/kernel-only diagnostics are not used as headline speedups.
The measured implementation passed 70 targeted tests, including CUDA.
Post-cleanup verification is recorded in [VALIDATION.md](../VALIDATION.md). Original source
files and prior artifacts are preserved.

## What is established and what remains unproven

Established for the measured subsets: original medical-format bytes can be
interpreted on the GPU correctly, and a compact GPU path can reduce completed
batch loading latency against efficient CPU alternatives. Bit unpacking and
window/layout conversion are distinct useful operations. Batching and source
record reuse matter, and CPU fallback is necessary for small requests.

Unproven: cold disk/network/GPU-direct I/O gains, end-to-end training gains for
the new native decoders, representative clinical-model performance, other GPUs,
mixed rates/skew/discontinuities/multi-segment records, arbitrary channel counts
for 212, FLAC/other entropy decoding, and a universal crossover threshold.
Studies use three recordings, warm OS caches and one laptop with unlocked
clocks. Repeated runs on that host are not independent datasets or hardware.

No clinical accuracy, diagnostic suitability or universal speedup is claimed.
The source-adapter restrictions are deliberate; unsupported semantics fail
instead of being silently normalized or resampled. Header compatibility quirks
and acquisition-range inconsistencies remain documented in the EDF reports.

## Concrete recommendation

The useful contribution is a small **native-format-aware GPU stage**, not a
general cache rewrite or GPU-only dataset framework. Keep metadata/source
validation on CPU, preserve original integer/packed bytes, and use a GPU stage
only where measured work amortizes transfer/launch cost. Retain strong CPU
implementations and numerical reference tests.

Before library integration, align the supported format subset and actual
consumer with maintainers, then validate that specific model/workload and storage
regime. This is a separate integration decision; it is not necessary to keep
adding benchmark repetitions or build a broad abstraction to complete the
current investigation. No external contact or integration submission occurred.

## Maintainer handoff (draft; not sent)

We investigated native EDF/EDF+ record assembly and native WFDB-212 decoding,
rather than only loading processed caches. A compact fused GPU 212 decoder
preserves signed samples, calibration and missing values against WFDB, including
full-record validation. It reduces warm source-to-GPU batch latency for larger
tested requests versus serial/four-thread compiled CPU controls, while the CPU
wins small requests. EDF record batching shows a similar workload-dependent
benefit. These results support optional source-aware GPU stages with CPU
fallback, not a universal medical NVDEC replacement. The next integration choice
should be driven by the actual format subset and training consumer.

All new work is uncommitted on `research/gpu-medical-timeseries`, based on
`cd5e3b259baede94fd659696b7d438ee35dfd524`. Reproduction commands, exact versions,
source/code hashes, per-round measurements and limitations accompany the
experiment documents. The user retains all Git and external submission actions.

The [research index](../README.md) lists active modules, current results and the
archive. Historical code hashes refer to the preserved source snapshot before
cleanup; current reports additionally hash the extracted shared helper module.
