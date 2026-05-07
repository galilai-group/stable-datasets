Something-Something V2
=====================

Something-Something V2 is a video action-recognition dataset of short clips
showing people performing fine-grained actions with everyday objects.

The dataset is distributed by Qualcomm under a non-standard data license. The
builder uses the Qualcomm download package and follows the normal
``stable-datasets`` cache rules. On machines with small home quotas, set
``STABLE_DATASETS_CACHE_DIR`` before loading the dataset, or pass
``download_dir=`` and ``processed_cache_dir=`` explicitly.

Usage
-----

.. code-block:: python

   from stable_datasets.video import SomethingSomethingV2

   ds = SomethingSomethingV2(split="train")
   sample = ds[0]

   video = sample["video"]          # VideoRef
   label = sample["label"]          # int class id, or -1 if unavailable
   text = sample["text"]            # instantiated caption
   template = sample["template"]    # class template

For large local scratch volumes, set the cache root before launching Python:

.. code-block:: bash

   export STABLE_DATASETS_CACHE_DIR=~/scratch/stable-datasets

Local Data
----------

If the Qualcomm assets are already available locally, pass ``data_dir=``. The
directory may contain an extracted ``labels/`` directory plus ``videos/``, or
the original label/video archives.

.. code-block:: python

   ds = SomethingSomethingV2(
       split="validation",
       data_dir="/path/to/something-something-v2",
   )

Returned Columns
----------------

- ``video``: ``VideoRef``
- ``video_id``: string id from the annotation files
- ``video_filename``: source filename
- ``label``: integer class id, ``-1`` when not available
- ``text``: instantiated action text
- ``template``: normalized class template
- ``placeholders_json``: JSON list of placeholder objects
- ``split``: split name
