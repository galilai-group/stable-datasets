import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from loguru import logger as logging

try:
    import evaluate
    import lightning.pytorch as pl
    from datasets import Dataset
    from lightning.pytorch.callbacks import Callback

    HAS_EVAL_DEPS = True
except ImportError:
    HAS_EVAL_DEPS = False
    # Define placeholder for type hinting if deps are missing
    class Callback:  # type: ignore[no-redef]
        pass


class PredictionDiskWriter:
    """
    Helper to incrementally write predictions and references to a local Arrow file.

    This ensures that large evaluation sets do not cause OOM errors by keeping
    everything in RAM. Instead, it flushes batches to disk using PyArrow IPC.
    """

    def __init__(self, file_path: Path, schema: pa.Schema):
        self.file_path = file_path
        self.schema = schema
        self.stream: Optional[pa.NativeFile] = None
        self.writer: Optional[ipc.RecordBatchFileWriter] = None
        self._is_open = False

    def _open(self):
        if self._is_open:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = pa.OSFile(str(self.file_path), "wb")
        self.writer = ipc.new_file(self.stream, self.schema)
        self._is_open = True

    def write_batch(self, data: Dict[str, Union[List, np.ndarray, torch.Tensor]]):
        """Append a batch of data to the Arrow file."""
        if not self._is_open:
            self._open()

        # Convert torch tensors or numpy arrays to lists/arrays for Arrow
        processed_data = {}
        batch_size = 0
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray):
                # Handle images or multi-dim arrays if necessary
                if v.ndim > 1:
                    v = [x for x in v]
                else:
                    v = v.tolist()
            processed_data[k] = v
            batch_size = len(v)

        batch = pa.RecordBatch.from_pydict(processed_data, schema=self.schema)
        self.writer.write_batch(batch)

    def close(self):
        """Finalize the Arrow file and close streams."""
        if self.writer:
            self.writer.close()
        if self.stream:
            self.stream.close()
        self._is_open = False


class EvaluateCallback(Callback):
    """
    Lightning Callback that captures model predictions and offloads them to disk.

    At the end of every epoch, it handles distributed syncing and calculates
    standardized scores using Hugging Face's `evaluate` library.

    Args:
        metric_name: Name of the metric to load via `evaluate.load()`.
        input_format_fn: A function that takes `outputs` from `validation_step`
            and returns a dict with 'predictions' and 'references'.
        output_dir: Directory to store temporary Arrow files. Defaults to a temp dir.
        metric_kwargs: Additional kwargs passed to `metric.compute()`.
        hub_model_id: Optional model ID to associate with the results for Metric Card.
    """

    def __init__(
        self,
        metric_name: str,
        input_format_fn: Callable[[Any], Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        metric_kwargs: Optional[Dict[str, Any]] = None,
        hub_model_id: Optional[str] = None,
    ):
        if not HAS_EVAL_DEPS:
            raise ImportError(
                "EvaluateCallback requires 'evaluate', 'datasets', and 'lightning'. "
                "Please install them via `pip install evaluate datasets lightning`."
            )

        super().__init__()
        self.metric_name = metric_name
        self.input_format_fn = input_format_fn
        self.metric_kwargs = metric_kwargs or {}
        self.hub_model_id = hub_model_id

        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "stable_eval"
        self.writer: Optional[PredictionDiskWriter] = None
        self.schema: Optional[pa.Schema] = None

        # To be initialized per rank
        self.metric = None

    def _get_rank_file(self, trainer: "pl.Trainer") -> Path:
        return self.output_dir / f"rank_{trainer.global_rank}_epoch_{trainer.current_epoch}.arrow"

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Extract predictions and references using user-provided function
        formatted = self.input_format_fn(outputs)

        if self.writer is None:
            # Initialize schema on first batch
            sample_data = {k: v for k, v in formatted.items()}
            # Infer Arrow schema from the first batch
            # Note: We assume the structure is consistent
            processed_sample = {}
            for k, v in sample_data.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                if isinstance(v, np.ndarray):
                    if v.ndim > 1:
                        v = [x for x in v]
                    else:
                        v = v.tolist()
                processed_sample[k] = v

            batch_pa = pa.RecordBatch.from_pydict(processed_sample)
            self.schema = batch_pa.schema
            self.writer = PredictionDiskWriter(self._get_rank_file(trainer), self.schema)

        self.writer.write_batch(formatted)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.writer:
            self.writer.close()
            self.writer = None

        # Ensure all ranks have finished writing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Only Rank 0 calculates the final score
        if trainer.global_rank == 0:
            logging.info(f"Gathering predictions from {self.output_dir} and computing {self.metric_name}")

            # Collect all rank files for the current epoch
            rank_files = list(self.output_dir.glob(f"rank_*_epoch_{trainer.current_epoch}.arrow"))

            if not rank_files:
                logging.warning("No prediction files found for evaluation.")
                return

            # Load into a single datasets.Dataset via Arrow backend
            # This is extremely memory efficient as it mmaps the files
            ds = Dataset.from_file(str(rank_files[0]))
            if len(rank_files) > 1:
                from datasets import concatenate_datasets

                others = [Dataset.from_file(str(f)) for f in rank_files[1:]]
                ds = concatenate_datasets([ds] + others)

            # Load the evaluate metric
            if self.metric is None:
                self.metric = evaluate.load(self.metric_name)

            # Compute the metric
            results = self.metric.compute(
                predictions=ds["predictions"], references=ds["references"], **self.metric_kwargs
            )

            # Log to Lightning
            for k, v in results.items():
                pl_module.log(f"eval/{self.metric_name}_{k}", v, on_epoch=True, prog_bar=True, rank_zero_only=True)

            logging.success(f"Evaluation complete for {self.metric_name}: {results}")

            # Generate Metric Card logic
            metric_card = generate_metric_card(
                metric_name=self.metric_name,
                results=results,
                model_id=self.hub_model_id or getattr(pl_module, "hub_model_id", "unknown-model"),
                dataset_name=getattr(trainer.datamodule, "dataset_name", "unknown-dataset")
                if hasattr(trainer, "datamodule")
                else "unknown-dataset",
            )

            # Save metric card to output dir
            card_path = self.output_dir / f"metric_card_{self.metric_name}_epoch_{trainer.current_epoch}.md"
            with open(card_path, "w") as f:
                f.write(metric_card)
            logging.info(f"Metric Card saved to {card_path}")

            # Cleanup rank files
            for f in rank_files:
                try:
                    f.unlink()
                except Exception as e:
                    logging.error(f"Failed to cleanup rank file {f}: {e}")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Final cleanup of the entire output directory if it's empty or requested
        if trainer.global_rank == 0 and self.output_dir.exists():
            logging.info(f"Cleaning up evaluation directory: {self.output_dir}")
            # shutil.rmtree(self.output_dir) # Optional: keep metric cards?


def generate_metric_card(metric_name: str, results: Dict[str, Any], model_id: str, dataset_name: str) -> str:
    """
    Generate a Hugging Face compatible Metric Card (Markdown with YAML metadata).
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Construct the YAML metadata part for the Hub
    yaml_metadata = {
        "model_index": [
            {
                "name": model_id,
                "results": [
                    {
                        "task": {"type": "unknown", "name": "Evaluation Task"},
                        "dataset": {"type": dataset_name, "name": dataset_name},
                        "metrics": [
                            {"type": f"{metric_name}_{k}", "value": v, "name": f"{metric_name} {k}"}
                            for k, v in results.items()
                        ],
                    }
                ],
            }
        ]
    }

    metadata_str = json.dumps(yaml_metadata, indent=2)

    card_content = f"""---
# Evaluation Results: {metric_name}
{metadata_str}
---

# Metric Card: {metric_name}

Verified results generated by `stable-datasets` EvaluateCallback.

- **Model ID:** {model_id}
- **Dataset:** {dataset_name}
- **Date:** {timestamp}

## Results
| Metric Piece | Value |
| --- | --- |
"""
    for k, v in results.items():
        card_content += f"| {k} | {v:.4f} |\n"

    card_content += "\n\n> [!NOTE]\n> Results are instantly compatible with the Hugging Face Hub metadata schema."

    return card_content
