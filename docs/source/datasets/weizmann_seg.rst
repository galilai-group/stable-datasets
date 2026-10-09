Weizmann Segmentation Evaluation Database
=========================================

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Image%20Segmentation-blue" alt="Task: Image Segmentation">
   <img src="https://img.shields.io/badge/Images-200-green" alt="Images: 200">
   <img src="https://img.shields.io/badge/Use-Evaluation%20Only-orange" alt="Use: Evaluation Only">
   </p>

Overview
--------

The Weizmann Segmentation Evaluation Database is a benchmark for figure-ground segmentation: separating one or two salient objects from the background. It contains 200 natural images, most with a longer side of 300 pixels, each provided in color and in grayscale. Each image comes with up to three human segmentations. It is meant for evaluation only: there is no training split and there are no class labels.

- **one_object**: 100 images showing one object, each with 3 human segmentations
- **two_objects**: 100 images showing two objects, each with 1 to 3 human segmentations (87 with 3, 8 with 2, 5 with 1)

In the original files, annotators painted the objects red (first object) and blue (second object) on top of the image. stable-datasets converts each of these into a label map: 0 = background, 1 = first object, 2 = second object.

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
     - Color (RGB) image
   * - ``image_bw``
     - ``PIL.Image.Image``
     - Grayscale (L) version of the same image; the original benchmark was designed for grayscale
   * - ``image_id``
     - str
     - Original image name (e.g. ``"00000155"``)
   * - ``segmentations``
     - list[bytes]
     - One PNG-encoded label map per annotator; decode with ``decode_mask``

``decode_mask`` returns a 2-D ``uint8`` NumPy array with the same height and width as the image.

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.images import WeizmannSegmentation
    from stable_datasets.images.weizmann_seg import decode_mask

    # First run will download + prepare cache, then return the split
    ds = WeizmannSegmentation(split="one_object")  # or "two_objects"

    # If you omit the split (split=None), you get a dict with both splits
    ds_all = WeizmannSegmentation(split=None)

    sample = ds[0]
    print(sample.keys())  # {"image", "image_bw", "image_id", "segmentations"}

    # Decode the human segmentations: 0 = background, 1 = first object, 2 = second object
    masks = [decode_mask(b) for b in sample["segmentations"]]
    print(len(masks), masks[0].shape)  # e.g. 3 (225, 300)

Evaluation
----------

Since the dataset has no training split, methods are typically trained elsewhere (e.g. on BSDS500) and applied as-is. The standard score is the F-measure between a predicted segment and the human segmentations; see Alpert et al. (2007) and the official evaluation code on the dataset website.

References
----------

- Official website: https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/
- Download page: https://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/dl.html

Citation
--------

.. code-block:: bibtex

    @inproceedings{alpert2007image,
      title={Image Segmentation by Probabilistic Bottom-Up Aggregation and Cue Integration},
      author={Alpert, Sharon and Galun, Meirav and Basri, Ronen and Brandt, Achi},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2007}
    }
