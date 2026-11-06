Frame stores (``stable_datasets.framestore``)
=============================================

A *frame store* is a `lance <https://lancedb.github.io/lance/>`_ dataset of JPEG frames, one per row (video frames,
or any images), converted so that any subset of a frame's MCUs (its 8x8 or 16x16 pixel blocks) can be read and
decoded on its own. Conversion is lossless: every frame decodes to exactly the same pixels as before, and the
converter checks that frame by frame. Loaders then read frames back without copies from RAM, local disk or network
storage, and decode them on the GPU (nvJPEG) and the CPU (libjpeg-turbo) at once.

Use it when JPEG decoding or reading limits training: video models that read many frames per clip, small GPUs per
CPU, or data on network storage. A loader that needs only part of each frame (e.g. LeVJEPA's token loader in
stable-pretraining, which decodes only the MCUs under the patches it keeps) can read only those parts.

Install
-------

.. code-block:: bash

    pip install "stable-datasets[framestore]"

Wheels for Linux (x86_64, aarch64; glibc 2.17+) and macOS 11+ (arm64, x86_64) ship framestore's C helpers prebuilt,
with libjpeg-turbo linked in. Elsewhere, or from a source checkout, they are compiled on first use with the system's C
compiler and libjpeg-turbo headers (``apt install build-essential libjpeg-turbo8-dev``, ``brew install jpeg-turbo``
or ``conda install -c conda-forge compilers libjpeg-turbo``). Without either, framestore still works: re-coding runs
a numba port of the C fast path (same output, about 75% of its speed) and ``jpegtran`` for the rare frames that need
libjpeg, and it says so once. The GPU loaders need Linux, an NVIDIA GPU and torchvision with CUDA.

Converting a store
------------------

The store is a lance dataset with a binary column of JPEG frames (``frame`` by default) and, for video, an episode
column (``episode_idx``) whose rows are each video's frames in order.

.. code-block:: bash

    # what the store holds, from the frame headers; writes nothing
    python -m stable_datasets.framestore.convert /data/frames.lance --census-only

    # convert in place (resumable: run the same command again after an interruption)
    python -m stable_datasets.framestore.convert /data/frames.lance

    # or keep the source and write the converted store elsewhere (needs ~2.1x the store's size)
    python -m stable_datasets.framestore.convert /data/frames.lance --out /data/frames_fs.lance

.. code-block:: python

    from stable_datasets.framestore import convert_store

    convert_store("/data/frames.lance", memory="16GB")  # the command line's options, by name; raises ConvertError

What conversion does:

- **Re-codes every frame** with a restart marker after each MCU, so each MCU's bits can be found and decoded alone.
  Pixels do not change; each frame is decoded before and after and compared (``--verify``, on the GPU when there is
  one, the CPUs joining in when it falls behind).
- **Stores the frames as a blob column** and adds ``seg_len``: each frame's per-MCU segment lengths. Every other
  column and the row order are kept.
- **Census first.** Progressive frames re-code to baseline losslessly. Frames that cannot be converted at all
  (arithmetic-coded, 12-bit, CMYK, unreadable) make it stop before writing anything, unless
  ``--keep-unsupported``, which stores them unchanged, with no ``seg_len``.
- **In place, one fragment at a time,** into a sibling ``<store>.converting``: the store never takes more than its
  size plus about 1.1x its largest fragment on disk.

Resources
---------

A node runs one job at a time, so conversion takes a share of it: ``--cpu``, ``--memory`` and ``--gpu-memory``,
each a fraction of what is free (default ``0.8``) or an amount (``--cpu 16``, ``--memory 16GB``), at lower CPU and
I/O priority (``--nice``, default 10). It reserves its fixed needs first (lance's scan, the census, verification
decodes, the GPU decoder, measured), bounds the bytes of frames in flight by what is left, and refuses before
writing anything if the fixed needs do not fit. Two CPUs always stay free. The loaders take the same arguments
through :class:`~stable_datasets.framestore.Resources`.

Reading
-------

.. code-block:: python

    from stable_datasets.framestore import FrameStore, SharedClipLoader

    store = FrameStore("/data/frames.lance", index="shared", read="mmap")
    loader = SharedClipLoader("/data/frames.lance", batch=64, frames=16, stride=2, size=(224, 224))
    clips = next(loader)  # (64, 16, 224, 224, 3) uint8 on the GPU, valid until the next call

``index``: ``"shared"`` keeps one index of every frame's location and segment lengths per node, in ``/dev/shm``
(built once by the first process; on local disk when ``/dev/shm`` is too small, or under
``$FRAMESTORE_INDEX_DIR``); ``True`` keeps one per process;
``False`` asks lance per batch.

``read``, the read policy, depends on where the store is:

.. list-table::
   :header-rows: 1

   * - Store on
     - Policy
     - Why
   * - RAM (page cache), local NVMe it fits in
     - ``mmap`` (default)
     - zero-copy from the page cache
   * - local NVMe larger than RAM
     - ``pread``
     - one exact-size read per frame, from a thread pool, instead of page faults
   * - network storage (NFS)
     - ``spans``
     - one read per clip (a clip's frames are neighbouring rows): fewer, larger requests
   * - network storage, partial reads
     - ``ranges``
     - only the byte ranges a loader asks for (``FRAMESTORE_ODIRECT=1`` bypasses the page cache on NFS)

Measured on an H100 node with 23 vCPUs (loader alone): 9,313 clips/s from RAM with ``mmap`` and a shared index;
7,769 clips/s from a cold VAST NFS store with ``spans``.

Decoding
--------

:class:`~stable_datasets.framestore.SharedClipLoader` decodes in GPU server processes (nvJPEG; run the NVIDIA MPS
daemon, ``nvidia-cuda-mps-control -d``, so they share the GPU with training) and in CPU processes (libjpeg-turbo).
``decoders="nvjpeg:2,cpu:16"`` sets both counts; a kind without a count is sized from the node. nvJPEG decodes on the
GPU's NVJPG engines where it has them (measured on H100; A100, B200 and B300 have them too) and on the CUDA cores
otherwise.

Store format
------------

The converter records the format in the store's schema metadata (key ``framestore``, a JSON ``format`` version;
``stable_datasets/framestore/store.py`` describes each version). Readers refuse a store in a newer format than they
know. Stores converted before the version was recorded read as format 1.

Limits
------

- Baseline and progressive 8-bit JPEGs with Huffman coding. Others are refused, or stored unchanged with
  ``--keep-unsupported``.
- Frames whose layout the partial readers do not handle (unusual chroma sampling, or an episode whose frames differ
  in quantization tables, layout or size) are converted but decoded whole by loaders that splice.
- CPU paths: Linux and macOS. GPU loaders: Linux with an NVIDIA GPU.
