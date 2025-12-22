CLEVRER
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Video%20Reasoning-blue" alt="Task: Video Reasoning">
   <img src="https://img.shields.io/badge/Videos-20%2C000-green" alt="Videos: 20,000">
   <img src="https://img.shields.io/badge/Resolution-480x320-orange" alt="Resolution: 480x320">
   <img src="https://img.shields.io/badge/Format-MP4-lightgrey" alt="Format: MP4">
   </p>

Overview
--------

**CLEVRER** (CoLlision Events for Video REpresentation and Reasoning) is a diagnostic video dataset designed for systematic evaluation of computational models on temporal and causal reasoning tasks.

The dataset contains **20,000 synthetic videos** of moving and colliding objects (spheres, cubes, cylinders) with various colors and materials. Each video is **5 seconds long** with **128 frames** at resolution **480×320**.

CLEVRER includes four types of questions:

- **Descriptive**: e.g., "What color is the sphere?"
- **Explanatory**: e.g., "What's responsible for the collision?"
- **Predictive**: e.g., "What will happen next?"
- **Counterfactual**: e.g., "What if the red cube were removed?"

Split sizes:

- **Train**: 10,000 videos (index 0-9999)
- **Validation**: 5,000 videos (index 10000-14999)
- **Test**: 5,000 videos (index 15000-19999)

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``video``
     - ``Video``
     - MP4 video file (480×320, 128 frames)
   * - ``scene_index``
     - int
     - Unique scene identifier
   * - ``video_filename``
     - str
     - Original video filename (e.g., "video_00001.mp4")
   * - ``questions_json``
     - str
     - JSON string containing list of questions with answers
   * - ``annotations_json``
     - str
     - JSON string containing object properties and collision events

Questions JSON Structure
------------------------

Each question in ``questions_json`` contains:

.. code-block:: python

    {
        "question_id": 0,
        "question": "What color is the sphere?",
        "question_type": "descriptive",  # or explanatory, predictive, counterfactual
        "answer": "blue",
        "choices": [...]  # for multiple choice questions
    }

Annotations JSON Structure
--------------------------

The ``annotations_json`` contains:

.. code-block:: python

    {
        "object_property": [
            {"object_id": 0, "color": "blue", "material": "rubber", "shape": "sphere"},
            ...
        ],
        "collision": [
            {"frame_id": 19, "object_ids": [0, 1]},
            ...
        ],
        "motion_trajectory": [...]
    }

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    import json
    from stable_datasets.images.clevrer import CLEVRER

    # Load the train split
    ds = CLEVRER(split="train")

    sample = ds[0]
    print(sample.keys())  # {"video", "scene_index", "video_filename", "questions_json", "annotations_json"}

    # Parse questions
    questions = json.loads(sample["questions_json"])
    print(f"First question: {questions[0]['question']}")
    print(f"Answer: {questions[0]['answer']}")

    # Parse annotations
    annotations = json.loads(sample["annotations_json"])
    print(f"Objects in scene: {len(annotations.get('object_property', []))}")

**Working with Videos**

.. code-block:: python

    # Access video frames (requires torchcodec)
    video = sample["video"]
    frame = video.get_frame_at(0)  # Get first frame
    print(f"Frame shape: {frame.data.shape}")

Requirements
------------

Video decoding requires ``torchcodec``:

.. code-block:: bash

    pip install torchcodec

.. note::

    **Large Download Size**: The CLEVRER video files are very large (~12GB for train, ~6GB each for validation/test).
    This dataset uses ``wget`` with resume support (``-c`` flag) instead of the standard download method to handle
    these large files reliably. If a download is interrupted, it will automatically resume from where it left off.

Related Datasets
----------------

- **CLEVR**: Static image version for visual reasoning
- **GQA**: Real-world visual question answering
- **Something-Something**: Video action understanding

References
----------

- Official website: http://clevrer.csail.mit.edu/
- Paper: `CLEVRER: CoLlision Events for Video REpresentation and Reasoning (ICLR 2020) <https://arxiv.org/abs/1910.01442>`_
- License: CC0

Citation
--------

.. code-block:: bibtex

    @inproceedings{yi2020clevrer,
        title={CLEVRER: CoLlision Events for Video REpresentation and Reasoning},
        author={Yi, Kexin and Gan, Chuang and Li, Yunzhu and Kohli, Pushmeet and Wu, Jiajun and Torralba, Antonio and Tenenbaum, Joshua B},
        booktitle={International Conference on Learning Representations},
        year={2020}
    }
