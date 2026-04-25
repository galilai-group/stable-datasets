#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BACKENDS = ("hf", "pyarrow", "lance")
REGIMES = ("training", "sparse")
DECODE_MODES = ("on", "off")
DEFAULT_SCALING_LEVELS = (1, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    backend: str
    regime: str
    decode_mode: str
    experiment_name: str
    parallelism_level: int
    num_workers: int
    lance_cpu_threads: int | None
    lance_io_threads: int | None
    multiprocessing_context: str | None
    batch_size: int
    warmup_batches: int
    measured_batches: int
    replicate_id: int
    seed: int
    subset_fraction: float
    results_path: str
    profile_batches: int = 0
    profile_delay_batches: int = 0
    profile_output_dir: str = "profiling/results/profiles"
    profile_with_cprofile: bool = False


def concurrency_for_backend(
    backend: str,
    parallelism_level: int,
    experiment_name: str,
    *,
    lance_scaling_mode: str = "threads",
) -> tuple[int, int | None, str | None]:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}")
    if experiment_name not in ("baseline", "scaling"):
        raise ValueError(f"Unknown experiment_name: {experiment_name!r}")
    if lance_scaling_mode not in ("threads", "workers"):
        raise ValueError(f"Unknown lance_scaling_mode: {lance_scaling_mode!r}")

    if experiment_name == "baseline":
        if backend == "lance":
            if lance_scaling_mode == "workers":
                return 0, None, None
            return 0, 1, None
        return 0, None, None

    if backend == "lance":
        if lance_scaling_mode == "workers":
            return parallelism_level, None, "spawn"
        return 0, parallelism_level, None
    return parallelism_level, None, None


def iter_run_specs(
    *,
    dataset: str,
    backends: tuple[str, ...] = BACKENDS,
    regimes: tuple[str, ...] = REGIMES,
    decode_modes: tuple[str, ...] = ("on",),
    scaling_levels: tuple[int, ...] = DEFAULT_SCALING_LEVELS,
    include_baseline: bool = True,
    include_scaling: bool = True,
    replicates: int = 3,
    batch_size: int = 128,
    warmup_batches: int = 50,
    measured_batches: int = 200,
    seed: int = 1337,
    subset_fraction: float = 0.05,
    results_path: str = "profiling/results/raw_runs.csv",
    experiment_name_override: str | None = None,
    profile_batches: int = 0,
    profile_delay_batches: int = 0,
    profile_output_dir: str = "profiling/results/profiles",
    profile_with_cprofile: bool = False,
    lance_scaling_mode: str = "threads",
) -> list[RunSpec]:
    specs: list[RunSpec] = []

    for backend in backends:
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend!r}")
    for regime in regimes:
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime: {regime!r}")
    for decode_mode in decode_modes:
        if decode_mode not in DECODE_MODES:
            raise ValueError(f"Unknown decode_mode: {decode_mode!r}")

    for decode_mode in decode_modes:
        for regime in regimes:
            for backend in backends:
                if include_baseline:
                    num_workers, lance_cpu_threads, multiprocessing_context = concurrency_for_backend(
                        backend,
                        0,
                        "baseline",
                        lance_scaling_mode=lance_scaling_mode,
                    )
                    for replicate_id in range(replicates):
                        specs.append(
                            RunSpec(
                                dataset=dataset,
                                backend=backend,
                                regime=regime,
                                decode_mode=decode_mode,
                                experiment_name=experiment_name_override or "baseline",
                                parallelism_level=0,
                                num_workers=num_workers,
                                lance_cpu_threads=lance_cpu_threads,
                                lance_io_threads=None,
                                multiprocessing_context=multiprocessing_context,
                                batch_size=batch_size,
                                warmup_batches=warmup_batches,
                                measured_batches=measured_batches,
                                replicate_id=replicate_id,
                                seed=seed,
                                subset_fraction=subset_fraction,
                                results_path=results_path,
                                profile_batches=profile_batches,
                                profile_delay_batches=profile_delay_batches,
                                profile_output_dir=profile_output_dir,
                                profile_with_cprofile=profile_with_cprofile,
                            )
                        )

                if include_scaling:
                    for level in scaling_levels:
                        num_workers, lance_cpu_threads, multiprocessing_context = concurrency_for_backend(
                            backend,
                            level,
                            "scaling",
                            lance_scaling_mode=lance_scaling_mode,
                        )
                        for replicate_id in range(replicates):
                            specs.append(
                                RunSpec(
                                    dataset=dataset,
                                    backend=backend,
                                    regime=regime,
                                    decode_mode=decode_mode,
                                    experiment_name=experiment_name_override or "scaling",
                                    parallelism_level=level,
                                    num_workers=num_workers,
                                    lance_cpu_threads=lance_cpu_threads,
                                    lance_io_threads=None,
                                    multiprocessing_context=multiprocessing_context,
                                    batch_size=batch_size,
                                    warmup_batches=warmup_batches,
                                    measured_batches=measured_batches,
                                    replicate_id=replicate_id,
                                    seed=seed,
                                    subset_fraction=subset_fraction,
                                    results_path=results_path,
                                    profile_batches=profile_batches,
                                    profile_delay_batches=profile_delay_batches,
                                    profile_output_dir=profile_output_dir,
                                    profile_with_cprofile=profile_with_cprofile,
                                )
                            )

    return specs


def command_for_run(spec: RunSpec, python_executable: str = "python") -> list[str]:
    cmd = [
        python_executable,
        "profiling/benchmark.py",
        "--dataset",
        spec.dataset,
        "--backend",
        spec.backend,
        "--regime",
        spec.regime,
        "--decode",
        spec.decode_mode,
        "--experiment-name",
        spec.experiment_name,
        "--parallelism-level",
        str(spec.parallelism_level),
        "--batch-size",
        str(spec.batch_size),
        "--num-workers",
        str(spec.num_workers),
        "--warmup-batches",
        str(spec.warmup_batches),
        "--measured-batches",
        str(spec.measured_batches),
        "--replicate",
        str(spec.replicate_id),
        "--seed",
        str(spec.seed),
        "--subset-fraction",
        str(spec.subset_fraction),
        "--results-path",
        spec.results_path,
    ]
    if spec.lance_cpu_threads is not None:
        cmd.extend(["--lance-cpu-threads", str(spec.lance_cpu_threads)])
    if spec.lance_io_threads is not None:
        cmd.extend(["--lance-io-threads", str(spec.lance_io_threads)])
    if spec.multiprocessing_context is not None:
        cmd.extend(["--multiprocessing-context", spec.multiprocessing_context])
    if spec.profile_batches > 0:
        cmd.extend(["--profile-batches", str(spec.profile_batches)])
        if spec.profile_delay_batches > 0:
            cmd.extend(["--profile-delay-batches", str(spec.profile_delay_batches)])
        cmd.extend(["--profile-output-dir", spec.profile_output_dir])
    if spec.profile_with_cprofile:
        cmd.append("--profile-with-cprofile")
    return cmd


def shell_command_for_run(spec: RunSpec, python_executable: str = "python") -> str:
    return " ".join(shlex.quote(part) for part in command_for_run(spec, python_executable=python_executable))


def recommended_cpus_per_task(spec: RunSpec, slack: int = 2) -> int:
    if spec.backend == "lance":
        concurrency = max(spec.num_workers, spec.lance_cpu_threads or 0, spec.lance_io_threads or 0)
    else:
        concurrency = spec.num_workers
    return max(1, int(concurrency) + int(slack))


def slurm_job_name(spec: RunSpec) -> str:
    decode_tag = "dec-on" if spec.decode_mode == "on" else "dec-off"
    regime_tag = "train" if spec.regime == "training" else "sparse"
    parallel_tag = f"p{spec.parallelism_level}"
    return f"in1k-{spec.backend}-{regime_tag}-{decode_tag}-{spec.experiment_name}-{parallel_tag}-r{spec.replicate_id}"


def slurm_script_for_run(
    spec: RunSpec,
    *,
    repo_root: str,
    output_dir: str,
    partition: str = "batch",
    cpus_per_task: int | None = None,
    mem: str = "8G",
    time_limit: str = "12:00:00",
    venv_activate: str = ".venv/bin/activate",
    python_executable: str = "python",
) -> str:
    job_name = slurm_job_name(spec)
    cpus = cpus_per_task if cpus_per_task is not None else recommended_cpus_per_task(spec)
    out_base = Path(output_dir) / job_name
    command = shell_command_for_run(spec, python_executable=python_executable)

    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={out_base}_%j.out",
            f"#SBATCH --error={out_base}_%j.err",
            "",
            f"cd {repo_root}",
            f"source {venv_activate}",
            "",
            "set -a",
            "source .env 2>/dev/null || true",
            "set +a",
            "",
            "export STABLE_DATASETS_CACHE_DIR=/users/sboughan/scratch/.stable-datasets",
            "export HF_HOME=/users/sboughan/scratch/.stable-datasets/huggingface",
            "export HF_DATASETS_CACHE=/users/sboughan/scratch/.stable-datasets/huggingface/datasets",
            "",
            command,
            "",
        ]
    )


def submit_scripts(script_dir: Path) -> int:
    if not script_dir.exists():
        print(f"missing script dir: {script_dir}", flush=True)
        return 1

    paths = sorted(script_dir.glob("*.sbatch"))
    print(f"found={len(paths)}", flush=True)

    submitted = 0
    failed = 0
    for path in paths:
        proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
        if proc.returncode == 0:
            submitted += 1
            print(proc.stdout.strip() or f"submitted {path.name}", flush=True)
        else:
            failed += 1
            message = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
            print(f"FAILED {path.name}: {message}", flush=True)

    print(f"submitted={submitted} failed={failed}", flush=True)
    return 0 if failed == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate or launch ImageNet-1K profiling runs.")
    parser.add_argument("--dataset", default="ImageNet-1K", choices=("ImageNet-1K",))
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    parser.add_argument("--include-decode-off", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--parallelism-levels", nargs="+", type=int, default=list(DEFAULT_SCALING_LEVELS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--measured-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--subset-fraction", type=float, default=0.05)
    parser.add_argument("--results-path", default="profiling/results/raw_runs.csv")
    parser.add_argument("--experiment-name-override", default=None)
    parser.add_argument("--profile-batches", type=int, default=0)
    parser.add_argument("--profile-delay-batches", type=int, default=0)
    parser.add_argument("--profile-output-dir", default="profiling/results/profiles")
    parser.add_argument("--profile-with-cprofile", action="store_true")
    parser.add_argument("--lance-scaling-mode", choices=("threads", "workers"), default="threads")
    parser.add_argument("--mode", choices=("print", "run-local", "write-slurm", "submit"), default="print")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--slurm-dir", default="profiling/slurm/generated")
    parser.add_argument("--slurm-log-dir", default="profiling/slurm/logs")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--mem", default="8G")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--venv-activate", default=".venv/bin/activate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "submit":
        raise SystemExit(submit_scripts(Path(args.slurm_dir)))

    decode_modes = ("on", "off") if args.include_decode_off else ("on",)
    specs = iter_run_specs(
        dataset=args.dataset,
        backends=tuple(args.backends),
        regimes=tuple(args.regimes),
        decode_modes=decode_modes,
        scaling_levels=tuple(args.parallelism_levels),
        include_baseline=not args.skip_baseline,
        include_scaling=not args.skip_scaling,
        replicates=args.replicates,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        measured_batches=args.measured_batches,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        results_path=args.results_path,
        experiment_name_override=args.experiment_name_override,
        profile_batches=args.profile_batches,
        profile_delay_batches=args.profile_delay_batches,
        profile_output_dir=args.profile_output_dir,
        profile_with_cprofile=args.profile_with_cprofile,
        lance_scaling_mode=args.lance_scaling_mode,
    )

    if args.mode == "print":
        for spec in specs:
            print(shell_command_for_run(spec, python_executable=args.python_executable))
        return

    if args.mode == "run-local":
        for spec in specs:
            cmd = command_for_run(spec, python_executable=args.python_executable)
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
        return

    script_dir = Path(args.slurm_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    Path(args.slurm_log_dir).mkdir(parents=True, exist_ok=True)
    repo_root = str(Path(__file__).resolve().parent.parent)

    for spec in specs:
        script_text = slurm_script_for_run(
            spec,
            repo_root=repo_root,
            output_dir=str(Path(repo_root) / args.slurm_log_dir),
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time_limit,
            venv_activate=args.venv_activate,
            python_executable=args.python_executable,
        )
        script_path = script_dir / f"{slurm_job_name(spec)}.sbatch"
        script_path.write_text(script_text)
        print(script_path)


if __name__ == "__main__":
    main()
