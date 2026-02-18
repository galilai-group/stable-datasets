AWA2
====

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Zero--Shot%20Learning-blue" alt="Task: Zero-Shot Learning">
   <img src="https://img.shields.io/badge/Classes-50-green" alt="Classes: 50">
   <img src="https://img.shields.io/badge/Animals-50%20Species-purple" alt="Animals: 50 Species">
   </p>

Overview
--------

Animals with Attributes 2 (AWA2) is an image classification dataset featuring 50 animal classes, primarily used for attribute-based image recognition and zero-shot learning research. The dataset provides a rich collection of animal images across diverse species.

- **Test**: 37,322 images across 50 animal classes
- **Train**: N/A (test-only dataset)

The dataset is widely used for zero-shot learning tasks where models must recognize animal classes they haven't been trained on, using semantic attributes as a bridge between seen and unseen classes.

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
     - RGB image (variable dimensions)
   * - ``label``
     - int
     - Class label (0-49)

Animal Classes
--------------

The dataset includes 50 animal classes:

antelope, grizzly+bear, killer+whale, beaver, dalmatian, persian+cat, horse, german+shepherd, 
blue+whale, siamese+cat, skunk, mole, tiger, hippopotamus, leopard, moose, spider+monkey, 
humpback+whale, elephant, gorilla, ox, fox, sheep, seal, chimpanzee, hamster, squirrel, 
rhinoceros, rabbit, bat, giraffe, wolf, chihuahua, rat, weasel, otter, buffalo, zebra, 
giant+panda, deer, bobcat, pig, lion, mouse, polar+bear, collie, walrus, raccoon, cow, dolphin

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images.awa2 import AWA2

    # Load the test set (only split available)
    ds = AWA2(split="test")

    sample = ds[0]
    print(sample.keys())  # {"image", "label"}
    print(f"Label: {sample['label']}")  # Integer label (0-49)

    # Access the image
    image = sample["image"]
    print(f"Image size: {image.size}")  # Variable dimensions

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

**Getting Class Names**

.. code-block:: python

    from stable_datasets.images.awa2 import AWA2

    ds = AWA2(split="test")
    
    # Get class names
    class_names = ds.info.features["label"].names
    
    # Map label to class name
    sample = ds[0]
    class_name = class_names[sample["label"]]
    print(f"Animal: {class_name}")

**Typical Use Case: Zero-Shot Learning**

.. code-block:: python

    from stable_datasets.images.awa2 import AWA2

    ds = AWA2(split="test")
    
    # Define seen and unseen classes for zero-shot learning
    seen_classes = [0, 1, 2, 3, 4]  # Train on these
    unseen_classes = [5, 6, 7, 8, 9]  # Test on these
    
    # Filter dataset
    seen_data = ds.filter(lambda x: x["label"] in seen_classes)
    unseen_data = ds.filter(lambda x: x["label"] in unseen_classes)

References
----------

- Homepage: https://cvml.ista.ac.at/AwA2/
- Paper: Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly (IEEE TPAMI 2019)

Citation
--------

.. code-block:: bibtex

    @ARTICLE{8413121,
      author={Xian, Yongqin and Lampert, Christoph H. and Schiele, Bernt and Akata, Zeynep},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title={Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly},
      year={2019},
      volume={41},
      number={9},
      pages={2251-2265},
      keywords={Semantics;Visualization;Task analysis;Training;Fish;Protocols;Learning systems;Generalized zero-shot learning;transductive learning;image classification;weakly-supervised learning},
      doi={10.1109/TPAMI.2018.2857768}
