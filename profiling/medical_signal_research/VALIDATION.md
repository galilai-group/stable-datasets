# Cleanup validation — 2026-11-14

The cleanup extracts shared helpers, groups documentation/results, and preserves
legacy module entry points. It introduces no decoding or calibration algorithm
change. See the [research index](README.md) for the complete regression command.

## Automated regression suite

**74 passed in 8.38 seconds; no skips**, including real CUDA tests on the
NVIDIA RTX 4060 Laptop GPU. The nine test modules cover shared helper contracts,
Arrow windows, EDF metadata/calibration, neonatal validation, paired caches,
experiment orchestration, native EDF windows/records and WFDB-212 decoding.

New checks cover multiblock/empty-file hashing, completed-work CUDA timing and
fresh-process imports of both native benchmarks with legacy experiments,
PyArrow and pyEDFlib deliberately unavailable.

Ruff lint, formatting checks and `git diff --check` passed. All relative
documentation links were checked. No file present in the original repository
baseline `999dc58fd8f1eb5dc5bae0f455ed4a2555b2b0fe` was changed.

## Executable checks

All twelve commands completed successfully using existing local source files:

| Program | Verification configuration |
|---|---|
| `profiling.signal_loading` | CUDA, 3 windows × 128 samples × 2 channels |
| `profiling.edf_signal_loading` | PhysioNet EDF, CUDA, 3 × 128 |
| `profiling.neonatal_edf_validation` | Original neonatal EDF and expert labels, CUDA; explicit temporary normalized copy and out-of-range acceptance |
| `profiling.prepare_signal_cache` | Fresh paired cache from that validation report; explicit source-exception acceptance |
| `profiling.signal_training` | Five training paths, 3 steps, 1 repeat, 1 warmup, batch 3 × 128 |
| `profiling.signal_experiment` | One short comparison round, telemetry and four profiler traces |
| `profiling.native_edf_loading` | PhysioNet CPU and original neonatal CUDA, 3 × 128 each |
| `profiling.benchmark_edf_records` | Neonatal 4 × 128 and PhysioNet 32 × 1024; all seven paths |
| `profiling.benchmark_wfdb212` | 4 × 128 and 3 × 650,000 full-record frames; all six paths |

The full WFDB run again verified bit-exact digital decoding and matching
physical values/NaN masks against WFDB. Maximum physical error was
`1.6860663906714812e-10 V`, within the existing combined relative/absolute
tolerance. EDF record layouts matched native edfio samples exactly and all
physical outputs passed the MNE reference checks. The short training check
passed input, loss and updated-parameter comparisons.

These runs check functionality, not new performance conclusions: most use only
one round and one or two repetitions. All new reports, command arguments and
logs are temporary under `/tmp/medical-signals-cleanup-d_rhfep_/` and are not
required runtime inputs. Previous research measurements were not overwritten.

## Preservation checks

- All 100 moved files were verified byte-for-byte immediately after relocation.
  The five current research documents were then updated for the new layout;
  the other 95 files remain unchanged, including every measurement and trace.
- All 52 code-hash entries in the retained final benchmark reports match the
  source preserved in `archive/source_before_cleanup.tar.gz`.
- AST comparison confirms the executable bodies of the five extracted helper
  functions are unchanged. All three WFDB decoder modules are byte-identical
  to the pre-cleanup source.
- Original source recordings remain unchanged; the benchmarks check source
  hashes before and after execution.

The unrelated full repository suite, including dataset-download tests, was not
run. No staging, commit, push or external submission was performed.
