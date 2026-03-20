VPT Minecraft
=============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Imitation%20Learning-blue" alt="Task: Imitation Learning">
   <img src="https://img.shields.io/badge/Segments-~70k-green" alt="Segments: ~70k">
   <img src="https://img.shields.io/badge/Resolution-360p%20%40%2020Hz-orange" alt="Resolution: 360p @ 20Hz">
   <img src="https://img.shields.io/badge/Format-MP4%20%2B%20JSONL-lightgrey" alt="Format: MP4 + JSONL">
   <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT">
   </p>

Overview
--------

The **Video PreTraining (VPT)** contractor demonstrations dataset is a large collection of
human Minecraft gameplay recordings released by OpenAI. Each recording segment is up to
**5 minutes long** and pairs a compressed video observation (``.mp4``) with a matching
per-timestep action/state file (``.jsonl``).

Videos were captured at **720p** and downsampled to **360p at 20 Hz** to reduce storage.
The dataset spans multiple recorder versions covering different tasks:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Version key
     - Recorder version
     - Task
   * - ``v6``
     - 6.x
     - General free play (core recorder, feature-complete)
   * - ``v7``
     - 7.x
     - General free play with emphasis on early-game / tree chopping
   * - ``v8``
     - 8.x
     - House building from scratch (10-minute time limit)
   * - ``v9``
     - 9.x
     - House building from random starting materials (10-minute time limit)
   * - ``v10``
     - 10.0
     - Obtain diamond pickaxe (20-minute time limit)
   * - ``find_cave``
     - BASALT 2022
     - Find a cave (3-minute time limit)
   * - ``waterfall``
     - BASALT 2022
     - Build a waterfall (5-minute time limit)
   * - ``animal_pen``
     - BASALT 2022
     - Build a village animal pen (5-minute time limit)
   * - ``build_house``
     - BASALT 2022
     - Build a village house in style (12-minute time limit)

.. image:: teasers/minecraft_new.gif
   :align: center
   :width: 90%

.. image:: teasers/minecraft_house.gif
   :align: center
   :width: 90%

.. image:: teasers/minecraft_mine.gif
   :align: center
   :width: 90%

.. note::

   Video files are large (~50–100 MB per 5-minute segment). Use
   :py:meth:`~stable_datasets.video.vpt_minecraft.VPTMinecraft.fetch_video_chunk`
   for a lightweight availability check, or use
   :py:meth:`~stable_datasets.video.vpt_minecraft.VPTMinecraft.download_segment` /
   :py:meth:`~stable_datasets.video.vpt_minecraft.VPTMinecraft.download_segments`
   to cache files locally. Action JSONL files (~1–2 MB each) can also be fetched
   directly via :py:meth:`~stable_datasets.video.vpt_minecraft.VPTMinecraft.load_actions`
   without any local download.

Data Structure
--------------

``VPTMinecraft(version)`` returns a **list of segment descriptor dicts**. Each dict
contains:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Description
   * - ``relpath``
     - ``str``
     - Segment identifier: ``<recorder-version>/<alias>-<session-id>-<date>-<time>``
   * - ``video_url``
     - ``str``
     - Full URL to the ``.mp4`` video recording (360p, 20 Hz)
   * - ``action_url``
     - ``str``
     - Full URL to the ``.jsonl`` action file (one action dict per line)
   * - ``options_url``
     - ``str``
     - Full URL to the ``-options.json`` recorder configuration file
   * - ``checkpoint_url``
     - ``str``
     - Full URL to the ``.zip`` Minecraft world checkpoint

Downloading Files
-----------------

Use ``download_segment()`` or ``download_segments()`` to cache segments locally.
Files are stored under::

    {dest_dir}/vpt_minecraft/{recorder-version}/{filename}

where ``dest_dir`` defaults to ``~/.stable-datasets/downloads/``. Downloads are
**idempotent** (existing files are skipped) and written **atomically** (via a
temporary file renamed on completion) so interrupted downloads leave no corrupt files.
A ``filelock`` prevents corruption when multiple processes download in parallel.

The ``files`` parameter controls which file types are downloaded per segment:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Downloaded file
   * - ``"video"``
     - ``.mp4`` video recording (~50–100 MB per segment)
   * - ``"actions"``
     - ``.jsonl`` action file (~1–2 MB per segment)
   * - ``"options"``
     - ``-options.json`` recorder configuration (< 1 KB)
   * - ``"checkpoint"``
     - ``.zip`` Minecraft world checkpoint (varies)

``download_segments()`` runs downloads concurrently using a ``ThreadPoolExecutor``
(default 4 workers) and displays a ``tqdm`` progress bar over segments.

Action JSONL Structure
----------------------

Each line of an action file is a JSON object representing one timestep:

.. code-block:: python

    {
        "mouse": {
            "dx": 0.0, "dy": 0.0,          # mouse delta (scaled)
            "buttons": [], "newButtons": []
        },
        "keyboard": {
            "keys": ["key.keyboard.w"],     # currently held keys
            "newKeys": [],
            "chars": ""
        },
        "isGuiOpen": False,
        "hotbar": 4,                        # selected hotbar slot (0–8)
        "yaw": -112.35, "pitch": 8.1,      # camera angles
        "xpos": 841.36, "ypos": 63.0, "zpos": 24.95,
        "tick": 0,
        "milli": 1649575088006,
        "inventory": [
            {"type": "oak_planks", "quantity": 59},
            ...
        ],
        "stats": {
            "minecraft.custom:minecraft.jump": 4,
            ...
        }
    }

Usage Example
-------------

**List segments for a version**

.. code-block:: python

    from stable_datasets.video.vpt_minecraft import VPTMinecraft

    # Fetch the index for the obtain-diamond-pickaxe task
    segments = VPTMinecraft(version="v10")
    print(f"Total segments: {len(segments)}")   # ~6,000 five-minute clips

    seg = segments[0]
    print(seg["relpath"])      # e.g. "10.0/cheeky-cornflower-setter-02e496ce4abb-..."
    print(seg["video_url"])    # full .mp4 URL
    print(seg["action_url"])   # full .jsonl URL

**Iterate with a limit**

.. code-block:: python

    for seg in VPTMinecraft.iter_segments(segments, max_segments=10):
        print(seg["relpath"])

**Load actions for one segment**

.. code-block:: python

    actions = VPTMinecraft.load_actions(seg)
    print(f"Timesteps: {len(actions)}")         # up to ~6,000 at 20 Hz

    first = actions[0]
    print(first["keyboard"]["keys"])            # e.g. ["key.keyboard.w"]
    print(first["mouse"]["dx"], first["mouse"]["dy"])

**Parse a relpath into its components**

.. code-block:: python

    info = VPTMinecraft.parse_relpath(seg["relpath"])
    # {"version": "10.0", "contractor_alias": "cheeky-cornflower-setter",
    #  "session_id": "02e496ce4abb", "date": "20220421", "time": "092639"}

**Quick availability check (no full download)**

.. code-block:: python

    # Fetches only the first 4 KB via HTTP Range request
    chunk = VPTMinecraft.fetch_video_chunk(seg, num_bytes=4096)
    assert b"ftyp" in chunk[:12]   # valid MP4 header

**Download a single segment on demand**

.. code-block:: python

    # Downloads video + actions to ~/.stable-datasets/downloads/ by default.
    # Files already on disk are skipped automatically.
    paths = VPTMinecraft.download_segment(seg)
    print(paths["video"])    # PosixPath('~/.stable-datasets/downloads/vpt_minecraft/10.0/<name>.mp4')
    print(paths["actions"])  # PosixPath('~/.stable-datasets/downloads/vpt_minecraft/10.0/<name>.jsonl')

    # Download to a custom directory, actions only
    paths = VPTMinecraft.download_segment(seg, dest_dir="/data/vpt", files=("actions",))

**Download many segments in bulk (concurrent)**

.. code-block:: python

    segments = VPTMinecraft(version="v10")

    # Download the first 20 segments (video + actions) using 4 threads
    all_paths = VPTMinecraft.download_segments(segments, max_segments=20)
    for paths in all_paths:
        print(paths["video"], paths["actions"])

    # Download actions only for all segments with 8 workers
    all_paths = VPTMinecraft.download_segments(
        segments, files=("actions",), max_workers=8
    )

**Load all available versions**

.. code-block:: python

    for version in VPTMinecraft.INDEX_URLS:
        segs = VPTMinecraft(version=version)
        print(f"{version}: {len(segs)} segments")

Related Datasets
----------------

- **BASALT 2022**: Task-specific subsets (``find_cave``, ``waterfall``, ``animal_pen``,
  ``build_house``) collected for the `MineRL BASALT NeurIPS 2022 competition
  <https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition>`_.
- **MineRL**: The Minecraft reinforcement learning environment used as the basis for recording.
- :doc:`clevrer`: Another video dataset in stable-datasets (synthetic video reasoning).

References
----------

- Paper: `Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos <https://cdn.openai.com/vpt/Paper.pdf>`_
- Blog post: https://openai.com/blog/vpt
- GitHub: https://github.com/openai/Video-Pre-Training
- License: MIT

Citation
--------

.. code-block:: bibtex

    @article{baker2022video,
      title={Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos},
      author={Baker, Bowen and Akkaya, Ilge and Zhokhov, Peter and Huizinga, Joost and
              Tang, Jie and Ecoffet, Adrien and Houghton, Brandon and Sampedro, Raul and
              Clune, Jeff},
      journal={arXiv preprint arXiv:2206.11795},
      year={2022}
    }
