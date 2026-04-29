# stable-datasets

`stable-datasets` is a research-oriented dataset library for images, time series/audio, and video. It is built around a simple idea: dataset onboarding should be easy, storage should be explicit, and the object you load should already be usable in PyTorch-style training code.

The library ships dataset builders under `stable_datasets/images/`, `stable_datasets/timeseries/`, and `stable_datasets/video/`. On first load, a builder downloads raw assets, writes a processed cache, and returns a map-style `StableDataset` or `StableDatasetDict`. After that, repeated loads hit the processed cache directly.

## What This Library Is For

This library is aimed at research workflows where you want:

- one place to keep dataset provenance, download logic, and schema definitions
- one loading interface across modalities
- cache-backed datasets that can be reopened quickly
- explicit storage choices instead of hidden format decisions
- a path to specialized layouts when one modality needs them

The central design goal is that datasets and modalities should have intelligent store-time and access-time defaults, while still leaving room for specialized layouts when workloads call for them. Video is the motivating example.

## Installation

```bash
pip install -e .
pip install -e ".[dev,docs]"
```

By default, downloads and processed caches live under `~/.stable_datasets/`. You can override the root with:

```bash
export STABLE_DATASETS_CACHE_DIR=/data/stable_datasets
```

or pass `download_dir=` / `processed_cache_dir=` per dataset.

## Examples

The `examples/` directory contains small, runnable examples:

- [examples/images.py](./examples/images.py): image classification with `CIFAR10`
- [examples/timeseries.py](./examples/timeseries.py): audio/time-series loading with `AudioMNIST`
- [examples/video.py](./examples/video.py): video loading with `SSv2`, `VideoRef`, and `set_video_decode`

### Image example

```python
from stable_datasets.images import CIFAR10

ds = CIFAR10(split="train")
sample = ds[0]

print(sample.keys())          # dict_keys(["image", "label"])
print(type(sample["image"]))  # PIL.Image.Image
print(sample["label"])        # integer class id

ds_torch = ds.with_format("torch")
torch_sample = ds_torch[0]
print(torch_sample["image"].shape)  # C x H x W
```

### Time-series example

```python
from stable_datasets.timeseries import AudioMNIST

ds = AudioMNIST(split="train")
sample = ds[0]

print(sample.keys())          # series, label, speaker_id, ...
print(len(sample["series"]))  # channels

ds_np = ds.with_format("numpy")
np_sample = ds_np[0]
print(np_sample["series"].shape)
```

### Video example

```python
from stable_datasets import VideoDecodeConfig
from stable_datasets.video import SSv2

ds = SSv2(split="train", storage_format="lance")
sample = ds[0]

video_ref = sample["video"]
print(video_ref.path)
print(video_ref.extension)

decoded = ds.set_video_decode(
    VideoDecodeConfig(
        num_frames=16,
        sampling="uniform",
        decoder="torchcodec",
        output="torch",
        layout="TCHW",
    )
)
decoded_sample = decoded[0]
print(decoded_sample["video"].shape)
```

The important semantic point is that default video rows return a `VideoRef`, which is a storage-normalized handle. Decoding is a retrieval-time policy and not baked into the stored schema.

## Architecture

The library has four main layers:

1. `BaseDatasetBuilder`
   Each dataset defines provenance, schema, and example generation. Builders are responsible for sourcing raw data and yielding Python examples.

2. Cache writers
   The cache layer turns builder output into a processed on-disk representation. Most datasets use the generic row-per-example writers in `stable_datasets/cache.py`.

3. Storage backends
   Backends reopen processed caches and provide row access. The main layouts today are:
   - `arrow-shards`
   - `lance-rows`
   - `lance-video-frames`

4. `StableDataset`
   The dataset object presents a uniform map-style API, handles formatting, and optionally applies read-time video decoding.

The practical flow is:

```text
Builder -> cache writer -> backend -> formatter -> StableDataset
```

### Repository shape

- `stable_datasets/schema.py`
  Public schema surface plus dataset metadata/config types.
- `stable_datasets/features/`
  Feature implementations such as `Image`, `Video`, `Array3D`, `ClassLabel`, and `Value`.
- `stable_datasets/backends/`
  Physical storage layouts and read-side logic.
- `stable_datasets/cache.py`
  Cache writers, metadata, and cache opening.
- `stable_datasets/dataset.py`
  `StableDataset`, `StableDatasetDict`, and read-time video decode integration.

### Features and schema

The distinction between `schema.py` and `features/` is deliberate:

- `features/` owns modality behavior
  This is where encode/decode/format logic for concrete feature types lives.
- `schema.py` owns dataset description
  This is where dataset metadata types, `Features`, versioning, and the stable import surface live.

That split matters because adding a new modality should mostly mean adding a new feature implementation, not threading special cases through unrelated files.

## Video modality

Video is the modality where storage and retrieval semantics diverge the most, so the library treats it explicitly.

### Generic video storage modes

`Video` currently supports three storage modes:

- `Video(storage="path")`
  Stages the source video into cache-owned assets and stores a structured cell describing the cached file.
- `Video(storage="bytes")`
  Stores the raw video bytes inline in the cache, again with a structured metadata cell.
- `Video(storage="frames")`
  Uses a specialized Lance frame layout for segment-oriented access.

At a practical level:

- choose `path` when you want the cache to own the video files but still preserve a normal compressed-video workflow
- choose `bytes` when you want the cache to be self-contained and not depend on separate staged files
- choose `frames` when your workload is really “sample lots of short frame windows quickly”.

For `path` and `bytes`, reading a row returns a `VideoRef`. A `VideoRef` is intentionally storage-only:

- it gives you `.path` and `.bytes`
- it carries metadata like extension and media type
- it does not own decoder construction or frame sampling

Storage decides what is cached; retrieval decides how to decode it.

So the default access pattern for `path` and `bytes` is:

```python
sample = ds[0]
ref = sample["video"]

# low-level access
video_path = ref.path
video_bytes = ref.bytes
```

From there, users either decode explicitly with their library of choice, or ask the dataset to decode at read time with `set_video_decode(...)`.

### Read-time decoding

If you want decoded tensors, use `set_video_decode(...)` with `VideoDecodeConfig`:

```python
decoded = ds.set_video_decode(
    VideoDecodeConfig(num_frames=16, sampling="random")
)
frames = decoded[0]["video"]
```

This keeps decode policy out of the persisted schema and lets users swap decoders or supply custom decode functions without rebuilding caches.

Custom hooks are also supported:

- `decode_fn` for per-sample custom decode
- `decode_fn_batched` for worker-local batched decode inside `StableDataset.__getitems__`

### The frame-WebP Lance layout

`Video(storage="frames")` is the specialized path with a different logical dataset layout.

The writer:

- fully decodes each source video
- optionally resizes frames
- re-encodes each frame as WebP
- stores one Lance row per frame

The backend then exposes frame windows as samples, not original videos as samples. In other words:

- dataset length becomes “number of valid frame windows”
- `sample["video"]` is already a frame stack
- rows also carry `start_frame`, `frame_indices`, and related segment metadata

So `frames` mode points users toward a different access pattern:

```python
sample = ds[0]
frames = sample["video"]
start = sample["start_frame"]
indices = sample["frame_indices"]
```

This layout is aimed at workloads like video SSL or action recognition where repeated short-window random access is more important than preserving the original compressed video as the primary read unit. However, `Video(storage="frames")` has a larger upfront preparation cost than path or bytes, because the cache writer decodes each source video, optionally resizes frames, and re-encodes them as WebP. The payoff is much faster random access to short temporal windows at training time.


## Onboarding new datasets

To add a new dataset:

1. choose the right modality package under `stable_datasets/`
2. subclass `BaseDatasetBuilder`
3. define `VERSION` and `SOURCE`
4. implement `_info()`
5. implement `_generate_examples(...)`
6. add tests that cover metadata and at least one smoke path

The main contract is that `_info()` and `_generate_examples(...)` must agree exactly on field names and types.

## Running tests

```bash
pytest -q
```

For targeted work, run the relevant subset under `stable_datasets/tests/`.
