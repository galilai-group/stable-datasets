"""
Submitit script for launching supervised learning evaluations on a SLURM cluster.

This script submits jobs to evaluate models on multiple datasets from stable-datasets.
"""

import json
import os
from pathlib import Path

import submitit


def main(kwargs, job_dir):
    dataset = kwargs["dataset"]
    model = kwargs["model"]
    seed = kwargs["seed"]
    results_file = kwargs["results_file"]
    batch_size = kwargs.get("batch_size", 128)
    num_workers = kwargs.get("num_workers", 8)
    image_size = kwargs.get("image_size", 224)
    lr = kwargs.get("lr", 5e-4)
    weight_decay = kwargs.get("weight_decay", 0.02)
    max_epochs = kwargs.get("max_epochs", 100)
    wandb_entity = kwargs.get("wandb_entity", "samibg")
    wandb_project = kwargs.get("wandb_project", "finalized-stable-datasets")
    config_name = kwargs.get("config_name", None)

    # Set up the executor folder to include the job ID placeholder
    executor = submitit.AutoExecutor(folder=job_dir / "%j")

    # Define SLURM parameters - all use gpu partition
    mem_gb = 48
    timeout_min = 4320  # 3 days (should be enough for most datasets)
    gpus_per_node = 1
    partition = "3090-gcondo"

    executor.update_parameters(
        mem_gb=mem_gb,  # Memory allocation
        slurm_ntasks_per_node=1,  # Number of tasks per node
        cpus_per_task=6,  # Number of CPUs per task
        gpus_per_node=gpus_per_node,  # Number of GPUs to use
        nodes=1,  # Number of nodes
        timeout_min=timeout_min,  # Maximum duration in minutes
        slurm_partition=partition,  # Partition name
        slurm_qos="cs-3090-gcondo",  # QOS for 3090-gcondo
        slurm_job_name=f"supervised_{model.split('/')[-1]}_{dataset}{f'_{config_name}' if config_name else ''}_seed{seed}",  # Job name
        slurm_mail_type="ALL",  # Email settings
        slurm_mail_user="sami_bou_ghanem@brown.edu",  # Email address
    )

    # Build command
    script_path = Path(__file__).parent / "supervised.py"
    venv_prefix = "/oscar/home/sboughan/stable-datasets-pyarrow/.venv"
    pyarrow_repo = "/oscar/home/sboughan/stable-datasets-pyarrow"
    data_dir = "/oscar/home/sboughan/scratch/.stable-datasets"
    command = (
        f"source {venv_prefix}/bin/activate && "
        f"export PYTHONPATH={pyarrow_repo}:$PYTHONPATH && "
        f"export STABLE_DATASETS_CACHE_DIR={data_dir} && "
        f"python -u {script_path} "
        f"--dataset {dataset} "
        f"--model {model} "
        f"--batch_size {batch_size} "
        f"--num_workers {num_workers} "
        f"--image_size {image_size} "
        f"--lr {lr} "
        f"--weight_decay {weight_decay} "
        f"--max_epochs {max_epochs} "
        f"--seed {seed} "
        f"--wandb_entity {wandb_entity} "
        f"--wandb_project {wandb_project} "
        f"--results_file {results_file} "
        f"--force_rerun"
    )
    # Add config_name if provided
    if config_name is not None:
        command += f" --config_name {config_name}"

    # Submit the job
    job = executor.submit(os.system, command)
    config_str = f" (config: {config_name})" if config_name else ""
    print(f"Job submitted for {model}/{dataset}{config_str} with seed {seed}, job ID: {job.job_id}")


def job_completed(model, dataset, seed, results_file, hyperparams):
    """Check if the job has already completed by looking for result files."""
    results_path = Path(results_file)

    if not results_path.exists():
        return False

    try:
        with open(results_path) as f:
            results = json.load(f)

        model_name = model.split("/")[-1]
        dataset_name = dataset.lower()

        # Check if model exists in results
        if model_name not in results:
            return False

        # Check if dataset exists
        if dataset_name not in results[model_name]:
            return False

        dataset_results = results[model_name][dataset_name]

        # Check new format (entries list)
        if isinstance(dataset_results, dict) and "entries" in dataset_results:
            entries = dataset_results["entries"]
            for entry in entries:
                existing_hyperparams = entry.get("hyperparams", {})
                # Compare hyperparams (including config_name if present)
                if existing_hyperparams == hyperparams:
                    config_name = hyperparams.get("config_name")
                    config_str = f" (config: {config_name})" if config_name else ""
                    print(f"✓ Job already completed: {model}/{dataset}{config_str} (seed {seed})")
                    return True

        return False
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error reading results file: {e}")
        return False


ALL_DATASETS = [
    "ArabicCharacters",
    "ArabicDigits",
    "Beans",
    "CIFAR10",
    "CIFAR100",
    "Country211",
    "CUB200",
    "DTD",
    "EMNIST",
    "FashionMNIST",
    "FGVCAircraft",
    "Flowers102",
    "Food101",
    "HASYv2",
    "Imagenette",
    "MedMNIST",
    "NotMNIST",
    "RockPaperScissor",
    "STL10",
    "SVHN",
]

# Config names for datasets that require them
DATASET_CONFIGS = {
    "EMNIST": ["balanced"],
    "MedMNIST": ["pneumoniamnist"],
}

DEFAULT_HYPERPARAMS = {
    "image_size": 224,
    "batch_size": 128,
    "lr": 5e-4,
    "weight_decay": 0.02,
    "max_epochs": 100,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit supervised evaluation jobs")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to submit (default: all). E.g. --datasets Country211 CUB200",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Skip the local completion check and always submit",
    )
    args = parser.parse_args()

    # Create the directory where logs and results will be saved
    job_dir = Path("./submitit_supervised")
    job_dir.mkdir(parents=True, exist_ok=True)

    model_list = [
        "WinKawaks/vit-small-patch16-224",
    ]

    dataset_list = args.datasets if args.datasets else ALL_DATASETS

    seed_list = [42]
    results_file = "./supervised_results.json"

    # Create results file if it doesn't exist
    results_path = Path(results_file)
    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({}, f)

    # Calculate total jobs
    total_jobs = 0
    for dataset in dataset_list:
        if dataset in DATASET_CONFIGS:
            total_jobs += len(model_list) * len(DATASET_CONFIGS[dataset]) * len(seed_list)
        else:
            total_jobs += len(model_list) * len(seed_list)

    print(f"{'=' * 60}")
    print("Submitting Supervised Learning Evaluation Jobs")
    print(f"{'=' * 60}")
    print(f"Models: {len(model_list)}")
    print(f"Datasets: {len(dataset_list)}")
    print(f"Seeds: {len(seed_list)}")
    print(f"Total jobs: {total_jobs}")
    print(f"{'=' * 60}\n")

    submitted_count = 0
    skipped_count = 0

    for seed in seed_list:
        for model in model_list:
            for dataset in dataset_list:
                configs = DATASET_CONFIGS.get(dataset, [None])

                for config_name in configs:
                    hyperparams = {**DEFAULT_HYPERPARAMS, "seed": seed}
                    if config_name is not None:
                        hyperparams["config_name"] = config_name

                    should_submit = args.force_rerun or not job_completed(
                        model, dataset, seed, results_file, hyperparams
                    )
                    if should_submit:
                        kwargs = {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "results_file": results_file,
                            "config_name": config_name,
                            **DEFAULT_HYPERPARAMS,
                        }
                        main(kwargs, job_dir)
                        submitted_count += 1
                    else:
                        skipped_count += 1

    print(f"\n{'=' * 60}")
    print("Submission Summary")
    print(f"{'=' * 60}")
    print(f"Jobs submitted: {submitted_count}")
    print(f"Jobs skipped (already completed): {skipped_count}")
    print(f"{'=' * 60}")
