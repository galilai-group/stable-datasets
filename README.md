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
- **Define `SOURCE`** (or override `_source()`): use `DatasetSource(...)` with a homepage, citation, and `assets` mapping of split/asset names to `DownloadInfo(...)`.
- **Implement `_info()`**: define the dataset schema and metadata.
- **Implement `_generate_examples(...)`**: yield `(key, example_dict)` pairs. The values in `example_dict` must match the schema from `_info()`.
- **Keep examples faithful to the raw dataset**: not every dataset needs a single `label` column. Structured metadata, multilabel targets, and modality-specific fields are fine.

Minimal skeleton:

```python
from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.utils import BaseDatasetBuilder


class MyDataset(BaseDatasetBuilder):
    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://example.com",
        citation="TBD",
        assets={
            "train": DownloadInfo(url="https://example.com/train.zip"),
            "test": DownloadInfo(url="https://example.com/test.zip"),
        },
    )

    def _info(self):
        return DatasetInfo(
            features=Features({"x": Value("int32")}),
            supervised_keys=("x",),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        # read from data_path (zip/npz/etc), then yield examples
        yield "0", {"x": 0}
```

### Onboarding a new dataset

When adding a new dataset, the main contract to uphold is:

- `_info()` and `_generate_examples(...)` must agree exactly on field names and types.
- `SOURCE` should be the single source of truth for provenance and downloads.
- If the dataset has multiple downloadable assets, use named assets and override `_split_generators()` when needed.
- If the dataset has official folds or nonstandard task structure, preserve that structure in the builder rather than inventing a random train/test split.
- If the dataset is not naturally single-label classification, it is still fine to yield richer dictionaries instead of forcing a `label` field.

In practice, a good onboarding checklist is:

1. Pick the right module under `stable_datasets/images/` or `stable_datasets/timeseries/`.
2. Add a `BaseDatasetBuilder` subclass with `VERSION`, `SOURCE`, `_info()`, and `_generate_examples(...)`.
3. Use `DatasetSource` and `DownloadInfo` for all raw assets. `DownloadInfo` can include fallback URLs.
4. Make the yielded examples faithful to the raw data, but normalized enough to be consistent:
   - image datasets usually yield `{"image": ..., "label": ...}`
   - timeseries/audio datasets usually yield `{"series": ...}` plus labels and metadata when available
5. Export the builder from the relevant `__init__.py`.
6. Add at least a small metadata or smoke test for the new builder.

Good examples to copy from:

- Image classification: [stable_datasets/images/cifar10.py](stable_datasets/images/cifar10.py)
- Timeseries classification: [stable_datasets/timeseries/audiomnist.py](stable_datasets/timeseries/audiomnist.py)
- Structured timeseries dataset without a single label: [stable_datasets/timeseries/groove_MIDI.py](stable_datasets/timeseries/groove_MIDI.py)

### Custom cache locations

By default:

- Downloads: `~/.stable_datasets/downloads/`
- Processed Arrow cache: `~/.stable_datasets/processed/`

You can override the base cache directory globally with the `STABLE_DATASETS_CACHE_DIR` environment variable:

```bash
export STABLE_DATASETS_CACHE_DIR=/data/my_cache
```

Or override per-dataset when constructing:

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
