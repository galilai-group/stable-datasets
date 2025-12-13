<div align="center">

# stable-datasets

_Datasets implemented as HuggingFace `datasets` builders, with custom download & caching._

</div>

This is an under-development research project; expect bugs and sharp edges.

## What is it?

- Datasets live in `stable_datasets/images/` and `stable_datasets/timeseries/`.
- Each dataset is a HuggingFace `datasets.GeneratorBasedBuilder` (via `BaseDatasetBuilder`).
- Downloads use local custom logic (`stable_datasets/utils.py`) rather than HuggingFace’s download manager.
- Returned objects are `datasets.Dataset` instances (Arrow-backed), which can be formatted for NumPy / PyTorch as needed.

## Minimal Example

```python
from stable_datasets.images.arabic_characters import ArabicCharacters

# First run will download + prepare cache, then return the split as a HF Dataset
ds = ArabicCharacters(split="train")

# If you omit the split (split=None), you get a DatasetDict with all available splits
ds_all = ArabicCharacters(split=None)

sample = ds[0]
print(sample.keys())  # {"image", "label"}

# Optional: make it PyTorch-friendly
ds_torch = ds.with_format("torch")
```

### Building a dataset with `BaseDatasetBuilder`

Each dataset is a Hugging Face `datasets.GeneratorBasedBuilder` subclass that follows a simple convention:

- **Define `VERSION`**: bump when your builder output changes.
- **Define `SOURCE`** (or override `_source()`): provides at least `{"homepage": "...", "citation": "...", "assets": {"train": "...", "test": "...", ...}}`.
- **Implement `_info()`**: defines features/metadata.
- **Implement `_generate_examples(self, data_path, split)`**: yields `(key, example_dict)`; `data_path` is the downloaded artifact for that split.

Minimal skeleton:

```python
import datasets

from stable_datasets.utils import BaseDatasetBuilder


class MyDataset(BaseDatasetBuilder):
    VERSION = datasets.Version("1.0.0")
    SOURCE = {
        "homepage": "https://example.com",
        "citation": "TBD",
        "assets": {
            "train": "https://example.com/train.zip",
            "test": "https://example.com/test.zip",
        },
    }

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({"x": datasets.Value("int32")}),
            supervised_keys=("x",),
            homepage=self.SOURCE.get("homepage"),
        )

    def _generate_examples(self, data_path, split):
        # read from data_path (zip/npz/etc), then yield examples
        yield "0", {"x": 0}
```

### Custom cache locations

By default:

- Downloads: `~/.stable_datasets/downloads/`
- Processed Arrow cache: `~/.stable_datasets/processed/`

You can override both when constructing a dataset:

```python
ds = ArabicCharacters(
    split="train",
    download_dir="/tmp/stable_datasets_downloads",
    processed_cache_dir="/tmp/stable_datasets_processed",
)
```

## Installation

```bash
pip install -e .
# Optional (dev tools + tests + docs):
pip install -e ".[dev,docs]"
```

## Running tests

```bash
pytest -q
```

Some tests download data and may be slow. You can filter by markers:

- **Skip slow tests**: `pytest -m "not slow"`
- **Run only download tests**: `pytest -m download`

## Generating teaser figures

Use the `generate_teaser.py` script to create visual previews of datasets for documentation:

```bash
# Generate a teaser with 5 samples
python generate_teaser.py --name CIFAR10 --num-samples 5 --output docs/source/datasets/teasers/cifar10_teaser.png

# Generate and display (without saving)
python generate_teaser.py --name MNIST --num-samples 8

# Customize figure size
python generate_teaser.py --name CIFAR100 --num-samples 10 --figsize 2.0 --output cifar100.png
```

## Datasets

See the module lists under `stable_datasets/images/` and `stable_datasets/timeseries/`.
