CLEVR
=====

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Visual%20Reasoning-blue" alt="Task: Visual Reasoning">
   <img src="https://img.shields.io/badge/Images-100%2C000-green" alt="Images: 100,000">
   <img src="https://img.shields.io/badge/Questions-~865%2C000-orange" alt="Questions: ~865,000">
   <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" alt="License: CC BY 4.0">
   </p>

Overview
--------

**CLEVR** (Compositional Language and Elementary Visual Reasoning) is a diagnostic dataset for testing a broad range of visual reasoning abilities. It contains synthetic images of simple 3D objects — cubes, spheres, and cylinders — rendered in varying colors, materials, and sizes, alongside automatically generated question-answer pairs designed to probe specific reasoning skills with minimal dataset bias.

CLEVR questions span five reasoning types:

- **Attribute identification**: "What color is the large cube?"
- **Counting**: "How many objects are either small cylinders or red things?"
- **Comparison**: "Is the sphere the same size as the metal cube?"
- **Spatial relationships**: "What size is the cylinder that is left of the brown metal thing?"
- **Logical operations**: "Are there an equal number of large things and metal spheres?"

Split sizes:

- **Train**: 70,000 images · 699,989 questions · scene graphs · functional programs
- **Val**: 15,000 images · 149,991 questions · scene graphs · functional programs
- **Test**: 15,000 images · 14,988 questions (no answers or scene annotations)

.. image:: teasers/clevr_teaser.png
   :align: center
   :width: 90%

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - RGB rendered scene image (320×480 pixels)
   * - ``image_filename``
     - str
     - Original filename (e.g., ``"CLEVR_train_000000.png"``)
   * - ``image_index``
     - int
     - Zero-based index of the image within its split
   * - ``scene_json``
     - str
     - JSON string with the ground-truth scene graph (objects, attributes, spatial relations). Empty ``{}`` for the test split.
   * - ``questions_json``
     - str
     - JSON string containing the list of questions associated with this image. Test-split questions omit ``answer`` and ``program`` fields.

Scene JSON Structure
--------------------

``scene_json`` decodes to a dict with the following shape:

.. code-block:: python

    {
        "image_index": 0,
        "image_filename": "CLEVR_train_000000.png",
        "objects": [
            {
                "color": "blue",         # gray | blue | brown | yellow | red | green | purple | cyan
                "material": "rubber",    # rubber | metal
                "shape": "sphere",       # cube | sphere | cylinder
                "size": "large",         # small | large
                "3d_coords": [x, y, z],
                "pixel_coords": [x, y, z],
                "rotation": 315.0        # degrees
            },
            ...
        ],
        "relationships": {
            "left":   [[...], ...],  # adjacency lists: relationships["left"][i] = indices of objects left of objects[i]
            "right":  [[...], ...],
            "front":  [[...], ...],
            "behind": [[...], ...]
        },
        "directions": {
            "left": [x, y, z], "right": [x, y, z],
            "front": [x, y, z], "behind": [x, y, z],
            "below": [x, y, z], "above": [x, y, z]
        }
    }

Questions JSON Structure
------------------------

``questions_json`` decodes to a list of question dicts:

.. code-block:: python

    [
        {
            "image_index": 0,
            "image_filename": "CLEVR_train_000000.png",
            "question": "How many blue cubes are there?",
            "answer": "2",               # omitted in test split
            "question_family_index": 12,
            "program": [                 # omitted in test split
                {"function": "scene",        "inputs": [],  "value_inputs": []},
                {"function": "filter_color", "inputs": [0], "value_inputs": ["blue"]},
                {"function": "filter_shape", "inputs": [1], "value_inputs": ["cube"]},
                {"function": "count",        "inputs": [2], "value_inputs": []}
            ]
        },
        ...
    ]

Programs are stored as topologically sorted lists of functions. Each function may consume outputs from earlier functions (referenced by index in ``inputs``) and/or literal string values (in ``value_inputs``).

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    import json
    from stable_datasets.images.clevr import CLEVR

    # First run will download + prepare cache (~18 GB), subsequent runs load from cache
    ds = CLEVR(split="train")

    # Omit split to receive a DatasetDict with all available splits
    ds_all = CLEVR(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "image_filename", "image_index", "scene_json", "questions_json"}

**Inspecting Scene and Questions**

.. code-block:: python

    import json
    from stable_datasets.images.clevr import CLEVR

    ds = CLEVR(split="val")
    sample = ds[0]

    scene = json.loads(sample["scene_json"])
    print(f"Objects in scene: {len(scene['objects'])}")
    for obj in scene["objects"]:
        print(f"  {obj['size']} {obj['color']} {obj['material']} {obj['shape']}")

    questions = json.loads(sample["questions_json"])
    for q in questions[:3]:
        print(f"Q: {q['question']}")
        print(f"A: {q['answer']}")

**Test Split (no answers)**

.. code-block:: python

    from stable_datasets.images.clevr import CLEVR

    ds_test = CLEVR(split="test")
    sample = ds_test[0]

    # scene_json is an empty dict for the test split
    # questions lack "answer" and "program" keys
    import json
    questions = json.loads(sample["questions_json"])
    print(questions[0].keys())  # {"image_index", "image_filename", "question", "question_family_index"}

Related Datasets
----------------

- :doc:`clevrer`: CLEVRER — a video extension of CLEVR that adds temporal and causal reasoning questions over collision events

References
----------

- Official website: https://cs.stanford.edu/people/jcjohns/clevr/
- Paper (arXiv): https://arxiv.org/abs/1612.06890
- Dataset generation code: https://github.com/facebookresearch/clevr-dataset-gen
- License: `Creative Commons CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_

Citation
--------

.. code-block:: bibtex

    @inproceedings{johnson2017clevr,
        title     = {CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
        author    = {Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and
                     Fei-Fei, Li and Zitnick, C. Lawrence and Girshick, Ross},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        pages     = {2901--2910},
        year      = {2017}
    }
