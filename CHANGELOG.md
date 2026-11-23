# Unreleased

- caches are Lance datasets only: `storage_format=`, `STORAGE_FORMAT`, the Arrow IPC writer and `ArrowBackend` are
  removed (passing `storage_format=` raises a `TypeError`); `pylance` is a core dependency; in-memory datasets use
  `TableBackend`; `map` and `flatten_indices` write Lance caches. Existing Lance caches keep their paths; Arrow caches
  are rebuilt as Lance on first load
- frame stores (`stable_datasets.framestore`, extra `framestore`): lance datasets of JPEG frames converted, losslessly
  and in place, so any subset of a frame's MCUs can be read and decoded on its own
  - `python -m stable_datasets.framestore.convert` / `convert_store`: census, re-coding with a restart marker after
    every MCU (verified frame by frame), resumable in-place commits, a resource budget (`--cpu`, `--memory`,
    `--gpu-memory`; 80% of a node by default)
  - `FrameStore`: zero-copy reads (`mmap`, `pread`, `spans`, `ranges`) and a per-node shared index
  - `SharedClipLoader`: decoded clips on the GPU from nvJPEG server processes and CPU decoders at once
  - the store format is versioned in the schema metadata
- platform wheels (Linux x86_64/aarch64, macOS arm64/x86_64) with framestore's C helpers prebuilt and libjpeg-turbo
  linked in; other platforms build them from the sdist, compile them on first use, or fall back to numba
- video storage (`video_storage=` on every builder, `VideoStorage`, `Video(storage="mp4" | "jpeg")`): the video
  column built as a framestore store whose samples are clip windows (`set_video_decode`), with `ds.loader()` for the
  GPU fast path. MP4 (default): one H.264 file per video re-encoded for clip reads (a keyframe every 4 frames, one
  coded size, the GPU-decodable subset) and decoded on the GPU by `framestore.gpu_h264`; JPEG: restart-coded frames
- `framestore.gpu_h264`: H.264 (the stores' subset) decoded by Triton kernels, bit-exact with ffmpeg;
  `framestore.h264`, the reference decoder; MP4 stores in `SharedClipLoader` (`decoders="triton:2,cpu"`)
- docs: Frame stores guide; Video storage guide

# V0.0.0

- file locking for concurrent downloads
- download recovery from partially downloaded files
- datasets:
  - CIFAR10/CIFAR10C
  - CIFAR100/CIFAR100C
  - KMNIST
  - MedMNISTv2
  - FashionMNIST
  - Food101
  - FGVCAircraft
  - TinyImagenet/TinyImagenetC
