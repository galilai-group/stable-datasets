# Reproduce on a fresh NVIDIA/Linux machine

This guide replaces reliance on dependencies and recordings left under the
original laptop's `/tmp`. Run from the repository root, on branch
`research/gpu-medical-timeseries`. The pre-refactor measured implementation is `b852c549`;
record your actual revision with `git rev-parse HEAD` when reproducing it.
On a cluster, use an allocated GPU job rather than a login node.

## Environment

The measured host used Linux x86_64, Python 3.13.9 and an RTX 4060 Laptop GPU.
The preserved training protocol records Linux 6.8.0-38 and NVIDIA driver
590.48.01. These observations are historical provenance, not a complete
compatibility specification for the recorded PyTorch CUDA build. The CPU model
was not recorded; do not infer it or fill historical reports with this Mac's CPU.

`requirements.txt` pins the recorded native-reader/numerical packages and the
reported PyTorch and Triton builds. These are top-level pins, not a fully resolved transitive lock.
The original pytest version was not recorded. Fresh installation and CUDA
execution of this recipe have not been verified on the Mac doing the handoff.

Use a separate environment. The following targets the recorded CUDA build;
availability of those exact wheels and compatibility with the installed driver
must be checked on the NVIDIA machine:

```bash
python3.13 -m venv "$HOME/.venvs/stable-datasets-medical"
source "$HOME/.venvs/stable-datasets-medical/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.15.0+cu132' --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r profiling/medical_signal_research/requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

If the exact PyTorch build is unavailable or fails the CUDA check, stop that
reproduction attempt and recover its installation source/wheels from the original
working laptop. Do not silently substitute another build and describe its results
as an exact reproduction. For a deliberately different environment, select a
compatible build using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/),
retain a separate requirements file, and label the new measurements accordingly.
No package installation is required merely to inspect saved JSON.

## Recordings and checksums

Only two EEG recordings and one ECG recording are required. The fourth file is
the ECG header. Run this once into a fresh directory; do not overwrite an
existing source collection:

```bash
signal_repo_dir="$PWD"
signal_data_dir="$HOME/stable-datasets-medical-data"
mkdir "$signal_data_dir"
curl --fail --location --max-time 120 --output "$signal_data_dir/eeg4.edf" https://zenodo.org/records/2547147/files/eeg4.edf
curl --fail --location --max-time 120 --output "$signal_data_dir/S001R01.edf" https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf
curl --fail --location --max-time 120 --output "$signal_data_dir/100.hea" https://physionet.org/files/mitdb/1.0.0/100.hea
curl --fail --location --max-time 120 --output "$signal_data_dir/100.dat" https://physionet.org/files/mitdb/1.0.0/100.dat
(cd "$signal_data_dir" && sha256sum -c "$signal_repo_dir/profiling/medical_signal_research/sources.sha256")
```

Checksums come from the retained benchmark reports. A mismatch must be resolved
before comparing with those reports. Do not repair or normalize the recordings.
These sources cover the native-format study; earlier neonatal label/cache
experiments additionally need `annotations_2017.mat`, outside this contribution. They are not required for the native study.

## Correctness and machine provenance

Create a fresh results directory outside the repository so previous measurements
remain unchanged. The hardware capture is best-effort and read-only; confirm it
contains the CPU model, driver and expected packages before sharing results.

```bash
signal_results_dir="$HOME/stable-datasets-medical-results-$(date +%Y%m%d-%H%M%S)"
mkdir "$signal_results_dir"
python -m profiling.research_environment > "$signal_results_dir/environment.json"
python -m pip freeze > "$signal_results_dir/environment.freeze.txt"
git rev-parse HEAD > "$signal_results_dir/revision.txt"
git diff --stat > "$signal_results_dir/working-changes.txt"
```

Run the retained tests (synthetic temporary fixtures, no dataset downloads):

```bash
python -m pytest -q -rs \
  stable_datasets/tests/test_signal_calibration.py \
  stable_datasets/tests/test_benchmark_support.py \
  stable_datasets/tests/test_edf_source.py \
  stable_datasets/tests/test_edf_record_loading.py \
  stable_datasets/tests/test_wfdb212.py
```

Check the pytest skip summary:
success with missing readers or skipped CUDA cases does not reproduce the
reported full GPU validation. Save the terminal output alongside the new results.
`environment.freeze.txt` records the actual resolved packages; it is not claimed
to reconstruct the unrecorded transitive environment of the original study.

## Native latency benchmarks

First run a short functional check, using fresh output files:

```bash
python -m profiling.benchmark_edf_records "$signal_data_dir/eeg4.edf" --batch-size 4 --length 128 --rounds 1 --repeats 1 --output "$signal_results_dir/edf-smoke.json"
python -m profiling.benchmark_wfdb212 "$signal_data_dir/100" --batch-size 4 --length 128 --rounds 1 --repeats 1 --output "$signal_results_dir/wfdb-smoke.json"
```

Then reproduce the EDF grid (defaults: seven rounds, thirty repeats):

```bash
for signal_source in eeg4 S001R01; do
  for signal_batch in 4 32; do
    for signal_length in 128 1024; do
      python -m profiling.benchmark_edf_records "$signal_data_dir/$signal_source.edf" --batch-size "$signal_batch" --length "$signal_length" --output "$signal_results_dir/$signal_source-b$signal_batch-l$signal_length.json"
    done
  done
done
```

This grouped invocation reproduces the configurations, not the original case
execution order. Individual path order remains balanced by the benchmark. The
original EDF order was 4×128, 32×1024, 4×1024, 32×128, alternating sources.

Reproduce the two WFDB workload series (defaults: six rounds, thirty repeats):

```bash
for signal_case in 4:128 32:1024 32:8192; do
  python -m profiling.benchmark_wfdb212 "$signal_data_dir/100" --batch-size "${signal_case%:*}" --length "${signal_case#*:}" --output "$signal_results_dir/wfdb-a-${signal_case/:/-}.json"
done
for signal_case in 32:8192 32:1024 4:128; do
  python -m profiling.benchmark_wfdb212 "$signal_data_dir/100" --batch-size "${signal_case%:*}" --length "${signal_case#*:}" --output "$signal_results_dir/wfdb-b-${signal_case/:/-}.json"
done
python -m profiling.benchmark_wfdb212 "$signal_data_dir/100" --batch-size 3 --length 650000 --rounds 1 --repeats 1 --output "$signal_results_dir/wfdb-full-record-check.json"
```

The full-record run checks correctness; its one-repeat timings are not a new
performance estimate. Both native benchmarks now append CPU/OS/package/driver
provenance to new reports after timing. Historical reports are retained as unchanged
objects in `results/edf.json` and `results/wfdb212.json`; their source hashes
refer to `results/measured-source.tar.gz`, not to the refactored implementation.
Report warm-source completed-batch latency separately from training performance,
cold storage and GPU-direct I/O. Changing hardware or drivers is a new experiment,
not a reason to expect identical latency.
