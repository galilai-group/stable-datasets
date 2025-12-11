<div align="center">

# stable-datasets

_Datasets implemented as HuggingFace `datasets` builders, with custom download & caching._

</div>


This is an under-development research project; expect bugs and sharp edges.

## What is it?

- Datasets live in `stable_datasets/images/` and `stable_datasets/timeseries/`.
- Each dataset is a HuggingFace `datasets.GeneratorBasedBuilder` (via `StableDatasetBuilder`).
- Downloads use local custom logic (`stable_datasets/utils.py`) rather than HuggingFace’s download manager.
- Returned objects are `datasets.Dataset` instances (Arrow-backed), which can be formatted for NumPy / PyTorch as needed.

## Minimal Example

```python
from stable_datasets.images.arabic_characters import ArabicCharacters

# First run will download + prepare cache, then return the split as a HF Dataset
ds = ArabicCharacters(split="train")

sample = ds[0]
print(sample.keys())  # {"image", "label"}

# Optional: make it PyTorch-friendly
ds_torch = ds.with_format("torch")
```

## Installation

```bash
pip install -e .
# Optional (dev tools + tests + docs):
pip install -e ".[dev,docs]"
```

## Cache layout

By default:
- Downloads: `~/.stable_datasets/downloads/`
- Processed Arrow cache: `~/.stable_datasets/processed/`

## Running tests

```bash
pytest -q
```

## Datasets

See the module lists under `stable_datasets/images/` and `stable_datasets/timeseries/`.
