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

## Small CNN diagnostic

Use the same prepared recording, and retain its published annotations outside
the repository. The MAT file is unchanged between dataset versions 3 and 4;
its published MD5 is `f0009a921d9dacd135fff4fc2377d872`.

```bash
curl --fail --location --max-time 120 --output "$signal_root/annotations_2017.mat" https://zenodo.org/records/4940267/files/annotations_2017.mat
```

Use `--subject-id 4` only with `eeg4.edf`. The command rejects a filename/subject
mismatch and annotations that do not cover the recording. Targets follow an
explicit experimental rule: a two-of-three expert majority per second, and a
positive window if any included second is positive. Starts and durations use
whole seconds; ambiguous subsecond label alignment is rejected. The label time
base is described in the [dataset paper](https://doi.org/10.1038/sdata.2019.39).

Run CUDA work in a Slurm allocation. For worker counts 0 and 2, request enough
CPUs for the two reader processes and the training process, initially four.
Keep each process's CPU pools bounded by setting `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1` and
`TOKIO_WORKER_THREADS=1`. Reader initialization also bounds Torch and Arrow
pools. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts for deterministic
CUDA model comparisons. Keep Slurm's GPU visibility unchanged.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m profiling.benchmark_signal_training "$signal_root/eeg4.edf" "$signal_root/prepared-1024" "$signal_root/annotations_2017.mat" --subject-id 4 --device cuda --batch-size 32 --steps 20 --workers 0 2 --trials 3 --output "$signal_root/cnn-workers.json"
```

Start with fewer steps/trials for a short functional test. The current-cache
baseline can dominate runtime; size the job's time limit using previous loader
measurements. For CPU development, use `--device cpu`; this runs the four CPU
paths and is not comparable to CUDA-endpoint measurements.

Results record the exact window starts/targets, first-batch MNE errors, every
step's loss, cumulative training time, loader-wait time, synchronized data-ready
time, compute time, worker startup, code/input hashes and allocation metadata.
The existing GPU path is run only in the training process with zero workers;
CPU paths use all requested counts. All trajectories must match within the
reported tolerance, or the command fails without writing a successful report.
Outputs must be outside the repository and must not already exist.

The CLI also creates `OUTPUT.progress.jsonl` (replacing the `.json` suffix).
It flushes and syncs metadata and each validated trial outside measurement.
On failure, the journal retains completed trials and a failure event; a final
successful report is written only after the whole comparison passes. A failed
trial's partial steps are diagnostic observations, not validated measurements.
Neither an existing report nor its progress journal is overwritten.

For a focused reproduction, add `--paths edf_memmap --workers 2 --trials 3`.
Keep the same batch size, steps, device and source as the failed run. Failure
context records whether the error occurred while starting the iterator, waiting
for a particular batch, preparing device data or executing the model. The
60-second worker timeout is unchanged; do not treat it as proof of a particular
NFS, worker-startup or pinning failure.

This single-recording diagnostic evaluates data stalls, not held-out seizure
accuracy. A later learning comparison must choose patient-separated splits and
validate the reference EEG model's stable-pretraining adaptation.
