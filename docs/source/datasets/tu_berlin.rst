TU-Berlin Sketch
================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Sketch%20Classification-blue" alt="Task: Sketch Classification">
   <img src="https://img.shields.io/badge/Classes-250-green" alt="Classes: 250">
   <img src="https://img.shields.io/badge/Size-~1111x1111-orange" alt="Image Size: ~1111x1111">
   </p>

  .. image:: teasers/tu_berlin_teaser.png
     :align: center
     :width: 90%

Overview
--------

The TU-Berlin dataset contains 20,000 human-drawn sketches evenly distributed over 250 everyday object categories
(80 sketches per category). Sketches were collected via crowdsourcing: participants were asked to draw an object
from a given category, and each sketch is stored as a 1111×1111 black-on-white drawing.

The original release does **not** define train/validation/test splits. This builder uses the splits of the public
Hugging Face re-upload `sdiaeyu6n/tu-berlin <https://huggingface.co/datasets/sdiaeyu6n/tu-berlin>`_, a random
(not class-stratified) 80% / 10% / 10% split made with ``train_test_split(random_state=42)``:

- **Train**: 16,000 images
- **Validation**: 2,000 images
- **Test**: 2,000 images

Because the split is random, the number of sketches per class varies slightly between splits. Images are converted to
single-channel grayscale. Results on this split are not directly comparable with papers that use their own splits or
the 3-fold cross-validation protocol from the original paper. The files are downloaded from the Hugging Face
repository, pinned to a fixed commit, because the original download is not reliably available.

Data Structure
--------------

When accessing an example using ``ds[i]`` you will receive a dictionary with these keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``image``
     - ``PIL.Image.Image``
     - Grayscale (mode ``L``) sketch, roughly 1111×1111
   * - ``label``
     - int
     - Class label (0-249), categories in alphabetical order

Usage Example
-------------

**Basic Usage**

.. code-block:: python

   from stable_datasets.images import TUBerlin

   # First run will download + prepare cache, then return the split
   ds = TUBerlin(split="train")

   # If you omit the split (split=None), you get a DatasetDict with all available splits
   ds_all = TUBerlin(split=None)

   sample = ds[0]
   print(sample.keys())  # {"image", "label"}

   # Optional: make it PyTorch-friendly
   ds_torch = ds.with_format("torch")

Related Datasets
----------------

- Natural-image classification benchmarks with similar class counts (see ``tiny_imagenet``, ``cifar100``)
- Other texture and shape datasets (see ``dtd``)

References
----------

- TU-Berlin project page: https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
- Hosted copy used by this builder: https://huggingface.co/datasets/sdiaeyu6n/tu-berlin
- Paper: M. Eitz, J. Hays, M. Alexa, "How Do Humans Sketch Objects?", ACM Trans. Graph. (SIGGRAPH) 31(4), 2012.

License: CC BY 4.0.

Citation
--------

.. code-block:: bibtex

   @article{eitz2012hdhso,
     author  = {Eitz, Mathias and Hays, James and Alexa, Marc},
     title   = {How Do Humans Sketch Objects?},
     journal = {ACM Trans. Graph. (Proc. SIGGRAPH)},
     year    = {2012},
     volume  = {31},
     number  = {4},
     pages   = {44:1--44:10}
   }