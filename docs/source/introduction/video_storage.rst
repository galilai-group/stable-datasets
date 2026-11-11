Video storage: MP4 or JPEG
==========================

Video datasets can be stored, when they are built, as a ``stable_datasets.framestore`` store whose samples are clip
windows, read fast from RAM, local disk or network storage and decoded on the GPU and the CPU at once. One argument
picks it:

.. code-block:: python

    from stable_datasets.schema import VideoStorage

    ds = SomethingSomethingV2(split="train", video_storage="mp4")                    # the defaults
    ds = SomethingSomethingV2(split="train", video_storage=VideoStorage(fps=7.5, short_edge=384))
    ds = SomethingSomethingV2(split="train", video_storage={"format": "jpeg", "fps": 15})

    clips = ds.set_video_decode(num_frames=16, resize=224)   # each sample: a 16-frame window
    loader = clips.loader(batch_size=64)                     # GPU fast path: (64, 16, 224, 224, 3) uint8 batches

The storage settings are part of the cache key, so each choice is its own cache, and a dataset never switches
format by itself (no size check decides it).

Which to choose
---------------

Measured on an H100 node (23 usable CPUs) with Kinetics-400 at 7.5 fps, 384 px short edge, the whole-frame loader
alone delivering 224 x 224 clips of 16 frames:

=========================  =================  ================  ==================================================
Store                      Size               Loader, clips/s   What it allows
=========================  =================  ================  ==================================================
``mp4`` (default)          1x (~the source)   4,131             compact; any decoder reads it (GPU, NVDEC, ffmpeg)
``jpeg``                   ~2x at the fps     3,369             partial decoding (token loader); any frame stride
=========================  =================  ================  ==================================================

On the CPU alone, JPEG decodes faster (MP4: about 600 clips/s on the same node). MP4 is decoded on the GPU's CUDA
cores by Triton kernels (``framestore.gpu_h264``, bit-exact with ffmpeg; needs Triton 3.6 or newer), with the CPU
joining in.

How an MP4 store is written
---------------------------

Each video is re-encoded once, when the dataset is built, so that reading a clip is cheap:

- **A keyframe every** ``keyint`` **frames** (4), closed GOPs, no B-frames: every window starts on a keyframe and
  decodes only its own frames. Windows start every ``keyint`` frames.
- **The frame rate is fixed when built** (``fps``): a clip's frames are consecutive stored frames, so the stride is
  chosen then (frameskip 4 at 30 fps: ``fps=7.5``). JPEG stores choose the stride when loading.
- **One coded size** (``canvas``, 16:9 at the short edge by default): narrower videos are padded with their edge,
  portrait ones transposed, wider ones scaled to fit; readers return every video in its own size and orientation.
  The GPU decoder and NVDEC reuse their state across videos of one size.
- **A subset of H.264 the GPU decodes in parallel**: baseline profile, one reference frame, a slice per macroblock
  row (each row decodes independently), no deblocking. About 7% more bytes than unconstrained settings.
- **crf** 18 by default (about 40 dB PSNR against the kept frames; ``crf=14``: ~41 dB, ~1.5x the bytes).

JPEG stores keep every frame as a JPEG with a restart marker after each MCU; see :doc:`framestore`.
