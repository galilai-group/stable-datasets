# Medical signal GPU research

Start with the [summary for Sami](SAMI_SUMMARY.md) or the
[detailed research conclusion](docs/RESEARCH_CONCLUSION.md).
The experiments investigate native EDF record assembly and WFDB-212 decoding;
they remain separate from the production dataset library.

## Current implementation

| Purpose | Modules in `profiling/` | Documentation |
|---|---|---|
| Shared calibration, validation, hashing and timing | `signal_utils.py` | Module and function docstrings |
| Native EDF source access and record assembly | `native_edf_loading.py`, `edf_record_loading.py`, `benchmark_edf_records.py` | [Source formats](docs/NATIVE_MEDICAL_FORMATS.md), [EDF implementation](docs/EDF_RECORD_LOADING.md), [workload grid](docs/EDF_WORKLOAD_LIMITS.md) |
| Native packed WFDB-212 decoding | `wfdb212.py`, `wfdb212_cpu.py`, `wfdb212_cuda.py`, `benchmark_wfdb212.py` | [WFDB methods, results and reproduction](docs/WFDB212_RESULTS.md) |

Command-line module names have not changed. Native experiments import the shared
helpers directly, without importing the earlier Arrow/cache experiments.
Legacy modules still re-export their previous helper names and remain runnable.

## Results and history

- `results/edf-workload-grid-v1/`: primary eight-case EDF study and summary.
- `results/wfdb212-final-{a,b}/` and `results/wfdb212-final-summary.json`: final WFDB runs.
- `results/wfdb212-full-record-*.json`: full-record correctness checks.
- [Archive](archive/README.md): earlier experiments, profiler traces, documentation,
  original-to-current path inventory, and the source snapshot before cleanup.

Historical measurement JSON remains byte-for-byte unchanged. Its code hashes
describe the measured source, preserved in the snapshot, rather than the later
helper extraction. New benchmark reports also hash `signal_utils.py`.
The cleanup validation is documented in [VALIDATION.md](VALIDATION.md).

## Run the regression tests

Use a CUDA-enabled PyTorch environment with NumPy, PyArrow, pytest, MNE,
pyEDFlib, edfio, SciPy, WFDB, Numba and Triton installed. In the existing local
setup, the extra medical readers are installed under `/tmp/stable-datasets-edf-deps`:

```bash
PYTHONPATH=/tmp/stable-datasets-edf-deps python -m pytest -q -rs \
  stable_datasets/tests/test_signal_utils.py \
  stable_datasets/tests/test_signal_loading_prototype.py \
  stable_datasets/tests/test_edf_signal_loading.py \
  stable_datasets/tests/test_neonatal_edf_validation.py \
  stable_datasets/tests/test_signal_cache.py \
  stable_datasets/tests/test_signal_experiment.py \
  stable_datasets/tests/test_native_edf_loading.py \
  stable_datasets/tests/test_edf_record_loading.py \
  stable_datasets/tests/test_wfdb212.py
```

CUDA tests require access to the GPU; check the skip summary. The full repository
suite also contains unrelated dataset downloads and is not this regression suite.
