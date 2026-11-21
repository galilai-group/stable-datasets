# Run the window-access experiment

Use the repository root on branch `research/gpu-medical-timeseries`. The source
is deliberately one neonatal EDF recording; this is not a full dataset build.
No Git mutation or external tracking/upload is performed by these commands.
On Hydra, execute CUDA work inside your allocated GPU job using the cluster's
account-specific instructions; no partition or NFS mount is assumed here.

## Environment

Use an existing compatible CUDA-enabled PyTorch environment, or a fresh Python
3.13 environment with the recorded package pins in `requirements.txt`. The
recorded torch 2.15.0+cu132 wheel source must be available and compatible with
the installed driver; a fresh installation of that exact CUDA environment has
not been verified on this Mac. Do not silently substitute versions when
claiming reproduction. Normal reference-reader dependencies must be installed
(not an incomplete `--no-deps` installation).

```bash
python -m pip install -r profiling/medical_signal_research/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
nvidia-smi
python -c "import torch, lance; print(torch.__version__, lance.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert hasattr(lance.LanceDataset, 'read_blob_ranges')"
```

For CPU-only development, CUDA-specific torch/Triton pins need not be installed.
Use a supported CPU PyTorch build plus Lance, PyArrow, NumPy, MNE, edfio,
pyEDFlib, pytest and matplotlib. Local compatibility checks use Python 3.10,
MNE 1.10.2, Lance 13.0.0 and PyArrow 25.0.1; those are separate from the
historical GPU environment. The benchmark requires file format 2.2 and the
planned blob-range API; it fails rather than silently loading whole blobs.

## Source and preparation

Choose paths **outside the repository**. On Hydra, set `signal_root` to your
chosen NFS research directory if that is the storage under study. Verify the
mount using your site's tools (for example `findmnt -T "$signal_root"` on Linux).
The `--storage-label` field merely records your description, not verification.

```bash
signal_root="$HOME/eeg-window-study"
mkdir -p "$signal_root"
curl --fail --location --max-time 120 --output "$signal_root/eeg4.edf" https://zenodo.org/records/2547147/files/eeg4.edf
```

Expected SHA256 for `eeg4.edf`:
`6444722cf46e101381754354888194dc1402ad445cd3da3df1df212e6365c81e`.
Check it before preparation (`sha256sum` on Linux or `shasum -a 256` on macOS).
Do not normalize or rewrite its header. The experiment uses edfio for source
metadata/native samples and MNE as the physical-value reference.

```bash
python -m profiling.benchmark_signal_windows prepare "$signal_root/eeg4.edf" "$signal_root/prepared-1024" --chunk-samples 1024
```

Preparation creates two immutable Lance stores, a production-format float32
recording-row cache, and metadata/provenance. It refuses an existing workspace.
At 256 Hz, 1024 samples is one four-second chunk; this is an initial choice,
not an optimized universal size. To compare chunk sizes, prepare a different
workspace and report its setup cost separately. Preparation may use substantial
RAM because it includes one whole recording and the production list encoder.

## Validate and measure

```bash
python -m pytest -q -rs \
  stable_datasets/tests/test_signal_calibration.py \
  stable_datasets/tests/test_benchmark_support.py \
  stable_datasets/tests/test_edf_source.py \
  stable_datasets/tests/test_edf_record_loading.py \
  stable_datasets/tests/test_lance_signal_windows.py
```

Inspect skips. Synthetic EDF fixtures and real temporary Lance datasets require
medical readers; CUDA checks require actual GPU access. Tests do not download
recordings. The paused WFDB study need not be run for this experiment.

CPU smoke run (four paths, CPU output):

```bash
python -m profiling.benchmark_signal_windows run "$signal_root/eeg4.edf" "$signal_root/prepared-1024" --device cpu --batch-sizes 1 4 --seconds 4 --repeats 1 --output "$signal_root/cpu-smoke.json"
```

Matched five-path GPU run and standalone figure:

```bash
python -m profiling.benchmark_signal_windows run "$signal_root/eeg4.edf" "$signal_root/prepared-1024" --device cuda --batch-sizes 1 4 16 32 --seconds 4 --repeats 10 --storage-label nfs --output "$signal_root/gpu-warm.json"
python -m profiling.benchmark_signal_windows plot "$signal_root/gpu-warm.json" --output "$signal_root/gpu-warm.png"
```

Use `--storage-label local` instead if measuring local storage. Reports/figures
refuse existing output filenames. All paths share the same seeded window
requests per batch size; every output is compared with MNE before timing.
Reports include errors, source/code hashes, preparation metadata, environment,
per-round latency and windows/second, and logical payload/materialization counts.
No cold-cache eviction is implemented and first reads after validation are warm.
Do not interpret selected-byte counters as measured physical storage bandwidth.

For the Hydra handoff, retain the generated preparation report, measurement
JSON, figure, environment and exact revision outside Git. Changing model size,
worker count, cache conditions or chunk size is a separate controlled comparison.
The subsequent training step must decide annotation-to-window labels and a
patient-separated split, select a small EEG model, validate its stable-pretraining
adaptation and compare matched learning progress versus elapsed wall time.
