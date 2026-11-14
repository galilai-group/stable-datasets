# Repeated training experiment and profiler diagnostics

## Question and fixed protocol

Does the integer-cache path consistently beat an already optimized float32
training path, and do transfers actually overlap model computation?

`signal_experiment.py` coordinates an explicit, recorded protocol. It never
changes power settings, GPU clocks, CPU governor or other processes. Before
running, AC power was online and no competing CUDA compute applications were
listed. This is a monitored laptop experiment, not a dedicated isolated host.

- Same neonatal paired cache v2, with C-contiguous float32/int16 arrays.
- Batch 32 x 1024 x 21; unchanged small Conv1D mean-regression workload.
- Two rounds, ten trials per path per round, 1,000 updates per trial.
- Fifty warmup updates before each timed trial; four PyTorch CPU threads.
- Resident, float32 serial, int16 serial, float32 overlap and int16 overlap.
- Same data seed, window schedule, initial model weights and SGD settings.
- Distinct order seeds (100 and 101) permute a base path order, then rotate it.
  Ten trials place each path in every position exactly twice per round.
- Setup and per-step scalar logging are excluded as in the previous experiment.
- Separate short profiler recordings: 20 updates with five warmups per path.
  Instrumented throughput is never substituted for uninstrumented throughput.

The GPU-resident reference stores the complete schedule on the GPU, so its
memory use scales with step count (about 2.6 GiB for the schedule here). These
tensors remain allocated in all paths. This reference is not a scalable loader.

## Measurements and artifacts

```bash
python -m profiling.signal_experiment /tmp/stable-datasets-neonatal/paired-cache-v2 --output-dir signal-reproducibility-v1 --steps 1000 --repeats 10 --rounds 2 --warmup 50 --cpu-threads 4 --profile-steps 20
```

Output is written to a **new** directory. A repeat requires a new name. The
runner writes its protocol before starting measurements and preserves completed
round reports if later stages fail. Existing results are never overwritten.

- `protocol.json`: settings, runner checksum, start host/power/GPU snapshot.
- `round-1.json`, `round-2.json`: training configuration, cache/source code
  checksums, every trial's position/repetition, Unix start/end timestamps,
  elapsed time, throughput, final loss and peak allocated GPU memory.
- `gpu-telemetry.csv`: one nvidia-smi process samples timestamp, GPU index,
  P-state, temperature, SM/memory clocks and GPU utilization every 500 ms.
- `gpu-telemetry.stderr`: monitoring errors, if any.
- `trace-{float32,int16}_{serial,overlap}.json`: Chrome/Perfetto-format traces.
- `summary.json`: both round summaries, descriptive telemetry ranges, end
  host snapshot and trace interpretation.

Telemetry runs only during uninstrumented measurement rounds, including setup.
The aggregate utilization statistic is not a per-trial or per-kernel measure.
Raw timestamps and recorded host timezone allow manual correlation with trial
boundaries. Power draw is deliberately not analyzed: the driver reports an
implausible value near 590 W on this laptop. Temperature/clock readings are
recorded as reported; no throttle cause is inferred solely from them.

## Correctness and interpretation

Cache opening checks all samples and hashes. Before trials, CUDA inputs,
first-step loss and updated parameters are checked across representations.
Round summarization checks every final loss against the resident reference.
The protocol and diagnostic regression tests additionally cover balanced order,
repeatability, interval unions and synthetic overlap traces.

For each round we report both the ratio of median int16/float32 overlap rates
and per-repetition block ratios. The blocks execute sequentially, not
simultaneously. Min/max ranges are descriptive, not confidence intervals;
trials on one warmed laptop need not be independent. No statistical
significance or universal speedup is inferred from these two rounds.

Trace parsing uses the `signal.measured_sequence` marker, excluding warmups and
model/buffer initialization. H2D intervals are unioned to avoid double counting.
CUDA kernels on transfer streams (including integer calibration) are excluded
from the compute-stream kernel union. The intersection reports the fraction
of observed H2D time that coincides with compute-stream kernels. It does not
measure the fraction of total training time saved, and does not attribute
kernel execution time to individual Python calls.

Named CPU ranges identify host-buffer waiting, gathering, H2D submission,
calibration submission, forward/target, backward and optimizer work. They
include Python/CUDA API waits and are not exclusive CPU usage. Do not add CPU
and GPU durations as if they were sequential. Profiler instrumentation can
alter scheduling and overlap; use traces for diagnosis, not speedup estimates.

## Validation commands

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs stable_datasets/tests/test_signal_loading_prototype.py stable_datasets/tests/test_edf_signal_loading.py stable_datasets/tests/test_neonatal_edf_validation.py stable_datasets/tests/test_signal_cache.py stable_datasets/tests/test_signal_experiment.py
```

All 44 tests passed in 5.20 s, with no skipped CUDA tests. A separate small smoke
experiment under `/tmp/signal-reproducibility-smoke-v1` verified telemetry and
real CUDA kernel/memcpy events before the longer experiment. Its short-run
throughput is not a research result. No additional dataset was downloaded.

Source provenance, accepted neonatal compatibility exceptions and the preceding
implementation remain documented in [SIGNAL_CACHE_TRAINING.md](SIGNAL_CACHE_TRAINING.md).

## Observations, 2026-11-14

Final artifacts are under `signal-reproducibility-v1/` at repository root.
Both measurement rounds completed, followed by four separate CUDA traces. All
final losses matched the resident reference within the prescribed tolerance.
The source remains the single neonatal recording, with the same explicit source
compatibility exceptions; there is no new evidence for native GPU EDF parsing.

| Path | Round 1 median steps/s | Round 2 median steps/s |
|---|---:|---:|
| GPU resident reference | 990.3 | 974.9 |
| Float32 serial | 534.7 | 524.6 |
| Int16 serial | 591.1 | 587.4 |
| Float32 overlap-enabled | 699.5 | 645.7 |
| Int16 overlap-enabled | 796.4 | 803.9 |

The optimized integer path beat the strongest measured float32 path by 13.9%
and 24.5% when comparing each round's medians. It was faster in 19 of 20 paired
repetition blocks. Block ratios ranged from 0.906 to 1.660; their medians were
1.211 and 1.199 for rounds 1 and 2. These statistics have different definitions
from a ratio of overall medians and are not interchangeable.

GPU telemetry recorded 327 samples at 500 ms intervals, including setup.
Temperature ranged from 43 to 71 C (median 64 C); SM clocks ranged from 2010
to 2460 MHz (median 2445 MHz), memory clocks from 7001 to 8001 MHz. Reported GPU
utilization ranged from 0 to 100%, with median 68%. AC remained online at both
snapshots. CPU governor was `powersave`, unchanged. Load averages were recorded
but the system was not isolated. These monitored conditions do not establish
fixed thermal/clock behavior or an externally controlled performance mode.

### Profiler observations, not throughput results

All four measured trace regions contained 20 H2D copies and 660 compute-stream
kernel events. Each region contains 20 updates; initialization and warmups are
excluded from these numbers.

| Trace | H2D union ms | H2D time intersecting compute kernels |
|---|---:|---:|
| Float32 serial | 5.734 | 0% |
| Float32 overlap-enabled | 5.800 | 0% |
| Int16 serial | 3.059 | 0% |
| Int16 overlap-enabled | 2.937 | 1.37% |

Thus an overlap-enabled code path is **not evidence of substantial realized
transfer/compute overlap**. In the recorded snippets there was little. The
enabled path also removes per-step synchronization; its uninstrumented speedup
must not be attributed solely to simultaneous copying/computation.

For the overlap-enabled traces, CPU gather-range durations totaled 8.310 ms
for float32 and 5.082 ms for int16. Int16 calibration-submission ranges added
2.113 ms. These ranges include API/Python overhead and do not identify exclusive
kernel cost. Reduced payload and host copying are supported as plausible
contributors, not a complete causal decomposition of the training advantage.

An offline union of all observed kernel, memcpy and memset intervals covered
23.58 of 35.34 ms (66.7%) for the float32 overlap-enabled snippet and 21.21 of
37.16 ms (57.1%) for int16. The remaining intervals are not automatically CPU
work: they may reflect host submission, waits or instrumentation effects. In
fact, the instrumented snippets made float32 appear faster than int16, unlike
the long uninstrumented trials. This directly demonstrates why profiler timing
must not replace the throughput results.

## Conclusion and next experiment

The integer cache's advantage recurred in both longer series for this small
model and recording, but its size varied. Storage/transfer volume is reduced;
large transfer/model overlap has not been demonstrated. Do not generalize the
result to other models, subjects, source formats or cold-storage workloads.

Next, inspect the traces for host submission and gathering stalls, and use a
lighter CUDA-event or Nsight experiment to distinguish profiler perturbation
from genuinely missing overlap. Then repeat with representative EEG compute
and additional recordings. Optimize the identified bottleneck rather than
building a custom decoder on the assumption that EDF parsing dominates.

## Extension protocol: twenty additional rounds

The user requested twenty full rounds instead of ten. The fixed extension uses
two consecutively launched blocks of ten rounds each, not independently dated
sessions. Every block receives ten seconds of the same GPU training workload
before measurement; every timed trial still receives fifty warmup updates with
fresh model weights. This fixed preparation does not guarantee thermal equilibrium.

All other workload settings remain: 1,000 steps, ten trials per path, batch
32 x 1024 x 21, four CPU threads. The new blocks therefore contribute 200 trials
per path. They are reported separately from the original two rounds, which did
not have this extra block preconditioning. No outcome-dependent stopping rule
or filtering of slower trials is applied.

```bash
python -m profiling.signal_experiment /tmp/stable-datasets-neonatal/paired-cache-v2 --output-dir signal-reproducibility-20-a --steps 1000 --repeats 10 --rounds 10 --warmup 50 --cpu-threads 4 --profile-steps 20 --order-seed-offset 0 --precondition-seconds 10
python -m profiling.signal_experiment /tmp/stable-datasets-neonatal/paired-cache-v2 --output-dir signal-reproducibility-20-b --steps 1000 --repeats 10 --rounds 10 --warmup 50 --cpu-threads 4 --profile-steps 20 --order-seed-offset 10 --precondition-seconds 10
```

Block A uses balanced order seeds 100..109; block B uses 110..119. Data/model
seed remains zero in every round. Each output directory retains its own
protocol, telemetry, round reports and separate short profiler traces. Existing
directories are preserved. A new regression check covers preconditioning and
the order-seed offset; all 45 tests passed in 3.30 s before starting the extension.

### Completed twenty-round results

Both blocks completed in full. Artifacts are under `signal-reproducibility-20-a/`
and `signal-reproducibility-20-b/`. The combined machine-readable analysis is
`signal-reproducibility-20-summary.json` at repository root. The original two
rounds are not pooled into these statistics.

Validation confirmed 20 complete rounds with 200 trials per path, balanced
position counts, matching cache/model/window-schedule/library/code signatures,
and one million timed optimizer updates across all five paths. Every final
training loss was exactly identical in the saved reports (maximum difference
zero). This is stronger than the tolerance requirement, but remains a numerical
systems check rather than a clinical accuracy result.

The primary statistic is the median of the twenty per-round ratios:
`median(int16_overlap steps/s) / median(float32_overlap steps/s)`.

| Statistic | Result |
|---|---:|
| Median throughput advantage across 20 round ratios | +14.7% |
| Middle 50% of round advantages (empirical quartiles) | +7.0% to +18.2% |
| Full range of round advantages | -6.1% to +49.6% |
| Rounds favoring int16 | 19 / 20 |
| Sequential repetition blocks favoring int16 | 153 / 200 |
| Block A median round advantage | +15.7% |
| Block B median round advantage | +11.1% |

Comparing against the best measured float32 path in each round, rather than
always its overlap-enabled variant, yields the same rounded median and
quartiles. One round's serial float32 median was marginally faster than its
overlap-enabled path; this does not change the rounded conclusion.

For reference, pooling all 200 trials per path gives the following medians.
Pooling trial rates is a different statistic from the primary round-ratio
summary and does not replace it.

| Path | Pooled median steps/s |
|---|---:|
| GPU resident reference | 978.7 |
| Float32 serial | 505.3 |
| Int16 serial | 550.3 |
| Float32 overlap-enabled | 643.1 |
| Int16 overlap-enabled | 733.6 |

The block-level pattern reproduces the direction of the advantage, but not a
fixed effect size or a guaranteed per-trial win. The 19/20 and 153/200 counts
are observed frequencies under this protocol, not estimated probabilities of
future success. No confidence interval is claimed: these blocks are consecutive
on the same host, and repeated windows/models are not independent datasets.

### Conditions and renewed profiling

Block A recorded 1,699 telemetry samples and block B 1,821. Temperatures ranged
43..68 C and 56..67 C respectively, with median 64 C in both blocks. Median SM
clock was 2445 MHz and median memory clock 8001 MHz in both blocks. AC was online
at every block boundary; the CPU governor remained `powersave`. System settings
were not changed, background activity was not excluded, and clocks were not
locked. The resident-reference trial rate itself ranged from 633.8 to 996.9
steps/s, demonstrating variation beyond the loader comparison.

The separate int16 overlap-enabled traces showed 19.9% of H2D time intersecting
compute kernels after block A and 1.31% after block B; float32 showed 0% in both
snippets. These are short instrumented observations, not the fraction of
training time saved. They reinforce that realized overlap is variable and
cannot by itself explain the long-run throughput difference. Int16 H2D unions
were again about half of float32 unions, while host gathering also used less
time in these recordings. Neither observation establishes a complete causal
decomposition of the advantage.

### Recommendation after twenty rounds

The evidence supports keeping int16 caching as an **optional** path for compatible
integer signals. Float32 should remain the reference/fallback, rather than being
replaced on the strength of this single recording and small model. Source-format
exceptions must remain explicit; no generic GPU EDF support has been established.

Further identical repetitions are now less informative than controlling the host
state, diagnosing host gathering/submission and profiling perturbation, and
testing a representative EEG model plus additional recordings. Validate that the
advantage survives those changes before proposing a library default or broad
performance claim. No commits, staging, pushes or external submissions occurred.
