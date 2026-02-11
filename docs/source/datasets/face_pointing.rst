Face Pointing
=============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Head%20Pose%20Estimation-blue" alt="Task: Head Pose Estimation">
   <img src="https://img.shields.io/badge/Persons-15-green" alt="Persons: 15">
   <img src="https://img.shields.io/badge/Format-JPEG-orange" alt="Format: JPEG">
   </p>

Overview
--------

The Head Pose Database consists of 15 sets of images, with each set containing 2 series of 93 images of the same person at different head poses. The database is used for head pose estimation and face orientation detection research.

- **Total Images**: 2,790 images across 15 persons
- **Format**: JPEG images
- **Size**: Approximately 30 MB
- **Annotations**: Each image filename encodes person ID, tilt angle, and pan angle

.. image:: teasers/face_pointing_teaser.png
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
     - RGB image of person's head at specific pose
   * - ``person_id``
     - int
     - Person identifier (1-15)
   * - ``angles``
     - List[int]
     - [tilt_angle, pan_angle] representing head orientation

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.face_pointing import FacePointing

    # Load the training split
    ds = FacePointing(split="train")

    # Access a single sample
    sample = ds[0]
    image = sample["image"]  # PIL Image
    person_id = sample["person_id"]  # 1-15
    angles = sample["angles"]  # [tilt, pan]

    print(f"Dataset size: {len(ds)}")
    print(f"Person {person_id}, Angles: {angles}")

**With PyTorch**

.. code-block:: python

    from stable_datasets.images.face_pointing import FacePointing
    import torch
    from torch.utils.data import DataLoader

    # Load dataset and format for PyTorch
    ds = FacePointing(split="train").with_format("torch")

    # Create a DataLoader
    dataloader = DataLoader(ds, batch_size=32, shuffle=True)

    for batch in dataloader:
        images = batch["image"]
        person_ids = batch["person_id"]
        angles = batch["angles"]
        # Your training loop here
        break

**Analyzing Head Poses**

.. code-block:: python

    from stable_datasets.images.face_pointing import FacePointing
    import numpy as np

    ds = FacePointing(split="train")

    # Collect angle statistics
    tilt_angles = []
    pan_angles = []
    
    for sample in ds:
        tilt, pan = sample["angles"]
        tilt_angles.append(tilt)
        pan_angles.append(pan)
    
    print(f"Tilt range: [{min(tilt_angles)}, {max(tilt_angles)}]")
    print(f"Pan range: [{min(pan_angles)}, {max(pan_angles)}]")

Dataset Details
---------------

- **Source**: Head Pose Image Database
- **Homepage**: http://crowley-coutaz.fr/HeadPoseDataSet/
- **Image Format**: JPEG, RGB color
- **Total Images**: 2,790 images
- **Persons**: 15 different individuals
- **Series per Person**: 2 series of 93 images each
- **Organization**: Files organized in directories by person
- **Special**: Includes a "Front" directory with 30 frontal images (pan and tilt angles equal to 0)

Angle Information
-----------------

Each image filename encodes the head pose angles:

- **Tilt angle**: Vertical head rotation (up/down)
- **Pan angle**: Horizontal head rotation (left/right)

Angles are encoded in the filename with the pattern: ``personXXYYY[+-]ZZ[+-]WW.jpg`` where:

- ``XX``: Person ID (01-15)
- ``YYY``: Image number within series
- ``[+-]ZZ``: Tilt angle in degrees
- ``[+-]WW``: Pan angle in degrees

Citation
--------

.. code-block:: bibtex

    @inproceedings{gourier2004estimating,
        title={Estimating face orientation from robust detection of salient facial features},
        author={Gourier, Nicolas and Hall, Daniela and Crowley, James L},
        booktitle={ICPR International Workshop on Visual Observation of Deictic Gestures},
        year={2004},
        organization={Citeseer}
    }

Related Datasets
----------------

- Head pose estimation datasets
- Face orientation datasets
- Facial landmark detection datasets
