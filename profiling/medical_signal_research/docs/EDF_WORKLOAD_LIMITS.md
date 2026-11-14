# EDF workload limits: batch size and window length

## Focus and protocol

This follow-up stays on the native-format question from Sami: which work should
run on CPU or GPU while loading the **same unchanged EDF sources**? No dataset,
format, decoder implementation or production dispatch mechanism was added.
The existing source-to-GPU benchmark was reused without code changes.

The earlier two corners changed batch size and window length together. This
2 x 2 grid separates those factors by adding few-long and many-short requests:

- Batch sizes: 4 and 32 windows.
- Window lengths: 128 and 1024 samples per channel.
- Sources: the same neonatal `eeg4.edf` and PhysioNet `S001R01.edf`.
- Seven source-to-GPU paths, seven balanced-order rounds, thirty measured
  repetitions and three warmups per path/round.
- All eight configurations were measured anew in this session. Earlier corner
  measurements were preserved and not mixed into this grid.

Both files have different channel counts and sample clocks: 21 at 256 Hz and
64 at 160 Hz. Equal sample lengths therefore do not imply equal physical
durations across files. Source readers and constant GPU metadata are prepared
outside timing. Window planning, original-source access, allocation, pinning,
transfer and the path's layout/calibration work are timed, ending at usable
GPU float32 values in volts. OS caches are warm. There is no training or cold
storage measurement and no direct GPU storage read.

The case execution order was 4x128, 32x1024, 4x1024, 32x128, alternating sources.
Paths are balanced within each case, but cases themselves are not independent
sessions or counterbalanced experiments. Hardware clocks were not locked.

## Artifacts and correctness

Results are under `profiling/medical_signal_research/results/edf-workload-grid-v1/`:

- `{neonatal,physionet}-b{4,32}-l{128,1024}.json`: all eight individual reports.
- `summary.json`: the comparison against the fastest measured non-GPU-assembly
  path at each point, individual round medians and record payload statistics.

All configurations validated bit-exact native CPU windows against edfio and
every physical GPU output against MNE float64 using the existing combined
tolerance (`rtol=2e-6`, `atol=1e-10 V`). Source digests were unchanged. Analysis
verified matching code digests, identical source identity per file, the expected
batch shapes, 7x30 measurements and every path occupying each position once.

The fastest non-GPU-assembly baseline may include GPU calibration; it is not a
CPU-only output. It represents CPU window assembly followed by the best measured
calibration/transfer alternative. Selection uses the lowest median of round
medians among the six such paths, rather than comparing only to a slow library
reader.

No source code changed, so previously passed unit tests were not redundantly
rerun. Correctness checks embedded in the benchmark ran before each of the eight
new measurement configurations.

## Results, 2026-11-14

Completed-batch latency in milliseconds; each value is the median of seven
round medians. Environment and source hashes match `EDF_RECORD_LOADING.md`.

| Batch x samples | Neonatal best CPU/hybrid | Neonatal GPU assembly | PhysioNet best CPU/hybrid | PhysioNet GPU assembly |
|---|---:|---:|---:|---:|
| 4 x 128 | 0.104 | 0.203 | 0.198 | 0.213 |
| 4 x 1024 | 0.228 | 0.239 | 0.450 | 0.293 |
| 32 x 128 | 0.420 | 0.320 | 0.675 | 0.398 |
| 32 x 1024 | 1.718 | 0.663 | 1.750 | 0.709 |

- **Few short windows:** direct CPU gathering and CPU calibration is fastest
  for both sources. The GPU path adds about 95% neonatal latency and 7.6%
  PhysioNet latency at these measured points.
- **Few long windows:** optimized CPU record assembly plus GPU calibration
  wins narrowly for neonatal (GPU assembly adds 4.8% latency). PhysioNet favors
  GPU assembly, with about 35% less latency than the strongest hybrid path.
- **Many short windows:** GPU assembly reduces latency by about 24% for
  neonatal and 41% for PhysioNet.
- **Many long windows:** GPU assembly reduces latency by about 61% and 59%.

For the selected baseline at each point, GPU assembly won all seven round
comparisons in the many-window cases and in PhysioNet's few-long case. It won
zero rounds for neonatal's few-short/few-long cases, and one of seven rounds
for PhysioNet's few-short case. These are correlated round observations, not
future-win probabilities or confidence intervals. Margins of 5–8% should not
be treated as robust universal decisions without further control of variability.

## Why byte count alone is insufficient

4 x 1024 and 32 x 128 have identical logical output sizes per source: 172,032
int16 bytes for neonatal, 524,288 for PhysioNet. Yet the preferred neonatal path
changes and both sources' absolute latencies differ substantially.

The actual record access also changes:

| Equal-output-size case | Neonatal record bytes | PhysioNet record bytes |
|---|---:|---:|
| 4 x 1024 | 150,528 | 474,720 |
| 32 x 128 | 462,336 | 825,600 |

Reading entire records for many short windows can transfer unused samples, while
overlapping windows may reuse records. These are logical record/transfer bytes,
not measured physical disk traffic. Request count, record overread and reuse
therefore vary together; this small grid does not causally isolate each one.
Channel count, sample rate and recording duration also differ between sources,
so their effects cannot be separated by comparing just these two files.

## Recommendation

The results support two candidates rather than a GPU-only loader: CPU assembly
for small requests and GPU record assembly for sufficiently batched workloads.
They do **not** justify a universal byte threshold or automatic routing rule.

The next targeted investigation should map the crossover more locally and
measure per-batch record payload/reuse, then check a representative consumer.
Packed or compressed medical formats remain a separate decoding question.
There is no reason to expand a generic dataset/cache framework to answer this
EDF source-format question.

## Reproduce

Use fresh output paths and the same isolated dependency environment:

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-neonatal/eeg4.edf --batch-size 4 --length 1024 --output new-neonatal-b4-l1024.json
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m profiling.benchmark_edf_records /tmp/stable-datasets-neonatal/eeg4.edf --batch-size 32 --length 128 --output new-neonatal-b32-l128.json
```

For all grid points, use both source paths and each combination of batch size
4/32 and length 128/1024. Default rounds=7 and repeats=30 match this study.
GPU access requires a normal terminal outside the restricted agent sandbox.
