# Physiological signal loading prototype

This is an isolated experiment, not a public library API or a native EDF reader.
It tests two opportunities identified in the current EEG builder:

1. Extract selected numerical windows from Arrow buffers without converting
   every signal sample into a Python float.
2. Transfer int16 samples and calibrate them on CUDA, compared with calibrating
   on CPU and transferring float32 samples.

No dataset download is needed. Existing builders, caches and formatting defaults
are unchanged. Input is deterministic synthetic data; results do not establish
real EEG loading or training speedups.

## Run locally

From the repository root with its existing dependencies installed:

```bash
python -m profiling.signal_loading --device cpu
python -m pytest -q stable_datasets/tests/test_signal_loading_prototype.py
```

## Run on the NVIDIA laptop, Hydra or Oscar

Copy the working tree, including the uncommitted prototype files, to the other
machine. A fresh clone will not include these files until you publish them.
Use a Python environment with NumPy, PyArrow, the repository's dependencies and
a CUDA-enabled PyTorch build compatible with the machine's NVIDIA driver.
Choose PyTorch installation instructions for that machine at
<https://pytorch.org/get-started/locally/>. A separate CUDA toolkit/compiler is
not required by this PyTorch-only prototype.

Check the environment:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

Then run:

```bash
python -m pytest -q stable_datasets/tests/test_signal_loading_prototype.py
python -m profiling.signal_loading --device cuda --batch-size 32 --length 1024 --channels 19 --repeats 50 > signal-loading-cuda.json
```

The default float32 batch is about 2.4 MiB, and its int16 counterpart is about
1.2 MiB. A 6 GB GPU is sufficient for these small inputs plus PyTorch runtime
allocations. Start with these defaults before increasing batch sizes. On a
cluster, run CUDA work in an allocated GPU job; use the account's documented
partition, GPU and time-limit settings rather than running it on a login node.
No cluster-specific resource names are assumed here.

## Interpret the results

- `cached_python_lists_to_cpu_tensor` uses the current formatter followed by
  dense NumPy/PyTorch conversion. Arrow row selection is prepared outside timing.
- `cached_numeric_windows_to_cpu_tensor` reads the same shuffled windows from
  two Arrow chunks, including request lookup and copying to writable float32.
  It supports sliced arrays, ragged recording lengths and repeated requests;
  selected windows must have uniform channels and no nulls.
- `prepared_float32_transfer` measures a completed transfer from prepared pinned
  memory. `prepared_int16_transfer_and_gpu_calibration` includes transfer and
  affine calibration, with calibration parameters already on the GPU. Input
  preparation is excluded and reported separately.
- `cpu_calibrate_pin_transfer` and `int16_pin_transfer_gpu_calibrate` include
  per-batch pinning and conversion/transfer work from the same CPU integer batch.
  These help expose a tradeoff that transfer-only timings would hide.
- CPU calibration is also reported separately. GPU timings synchronize before
  and after each call; they measure completed batch latency, not throughput of
  a pipelined/overlapped loader. Allocations in each callable are included.
- JSON records versions, dimensions, median/p95 latency, GPU name and peak
  PyTorch-allocated device memory (not total driver/device memory).
- Correctness is checked against a float64 CPU calculation before reporting.
  Float32 arithmetic uses tolerances, not bit-exact claims.

The physical units are whatever the supplied calibration metadata specifies.
This does not implement MNE's unit conventions, channel classification, mixed
sampling-rate handling, annotation alignment or EDF+ discontinuities. EDF sample
reading, disk/network access, storage-page decompression and actual model
training remain separate experiments. GPU calibration can lose to CPU for small
batches; reporting that result is useful.

## Next experiment

After validating the CUDA path on real hardware, choose a small genuine EDF
recording and compare window values/metadata against MNE. Measure preparation
and cached training separately before deciding whether to integrate a numerical
feature, a window backend, or a native-format GPU path into the library.
