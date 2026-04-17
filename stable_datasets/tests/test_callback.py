import os
import unittest
from pathlib import Path
import torch
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from stable_datasets.callbacks import PredictionDiskWriter, generate_metric_card

class TestCallbackUtils(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("test_eval_output")
        self.tmp_dir.mkdir(exist_ok=True)
        self.file_path = self.tmp_dir / "test.arrow"

    def tearDown(self):
        if self.file_path.exists():
            self.file_path.unlink()
        if self.tmp_dir.exists():
            for f in self.tmp_dir.glob("*"):
                f.unlink()
            self.tmp_dir.rmdir()

    def test_prediction_disk_writer(self):
        # Define a simple schema
        schema = pa.schema([
            ("predictions", pa.int64()),
            ("references", pa.int64()),
        ])
        
        writer = PredictionDiskWriter(self.file_path, schema)
        
        # Write some data
        data = {
            "predictions": torch.tensor([1, 0, 1]),
            "references": np.array([1, 1, 0])
        }
        writer.write_batch(data)
        writer.close()
        
        # Verify file exists and has content
        self.assertTrue(self.file_path.exists())
        
        with pa.memory_map(str(self.file_path), "r") as source:
            reader = ipc.open_file(source)
            batch = reader.get_batch(0)
            self.assertEqual(batch.num_rows, 3)
            self.assertEqual(batch.column(0).to_pylist(), [1, 0, 1])
            self.assertEqual(batch.column(1).to_pylist(), [1, 1, 0])

    def test_generate_metric_card(self):
        results = {"accuracy": 0.95, "f1": 0.94}
        card = generate_metric_card(
            metric_name="test_metric",
            results=results,
            model_id="test-model",
            dataset_name="test-dataset"
        )
        
        self.assertIn("accuracy", card)
        self.assertIn("test-model", card)
        self.assertIn("model_index", card)
        self.assertIn("---", card)

if __name__ == "__main__":
    unittest.main()
