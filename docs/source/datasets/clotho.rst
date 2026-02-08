Clotho
==========

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Audio%20Captioning-blue" alt="Task: Audio Captioning">
   <img src="https://img.shields.io/badge/Captions%20per%20Audio%20Sample-5-green" alt="Captions per Audio Sample: 5">
   <img src="https://img.shields.io/badge/Caption%20Length-8%20to%2020%20words-orange" alt="Caption Length: 8 to 20 words">
   </p>

Overview
--------

The Clotho dataset contains audio samples with a wide variety of general audio content, such as  wind blowing, TV static, paper shredding, etc. The total dataset contains 4,981 audio samples, and it has a "development" (train), "evaluation" (validation), and testing split. However, the testing split is not available here in ``stable-datasets``, as it is withheld by the dataset's creators for "for potential usage in scientific challenges". Every audio sample has exactly 5 captions, each ranging from 8 to 20 words, describing the audio content of the sample in free-form English text. Below are the dataset sizes:

- **Train**: 2,893 samples (14,465 total captions)
- **Validation**: 1,045 samples (5,225 total captions)

A sample audio clip from the dataset can be heard at ``docs/source/datasets/teasers/clotho_teaser.wav``.

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``audio``
     - ``torchcodec.decoders.AudioDecoder``
     - Audio clip loaded from a .wav file
   * - ``captions``
     - ``list[str]``
     - List of 5 captions
   * - ``keywords``
     - ``list[str]``
     - List of keywords relevant to the audio clip
   * - ``freesound_id``
     - ``int``
     - ID of original audio clip from Freesound
   * - ``freesound_link``
     - ``str``
     - URL to original audio clip from Freesound
   * - ``start_sample``
     - ``int``
     - Sample from the original Freesound audio that this audio clip starts from
   * - ``end_sample``
     - ``int``
     - Sample from the original Freesound audio that this audio clip ends at
   * - ``manufacturer``
     - ``str``
     - The Freesound user who published the audio
   * - ``license``
     - ``str``
     - The license that the audio is published under


Usage Example
-------------

.. code-block:: python

    from stable_datasets.timeseries.clotho import Clotho

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = Clotho(split="train")

    # Can access attributes of each sample through standard Python dict indexing
    sample = ds[0]
    print(sample.keys())
    print(f"Captions: {sample['captions']}")
    print(f"Keywords: {sample['keywords']}")

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Homepage: https://github.com/audio-captioning/clotho-dataset

Citation
--------

.. code-block:: bibtex

    @inproceedings{9052990,
        author={Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
        booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
        title={Clotho: an Audio Captioning Dataset}, 
        year={2020},
        volume={},
        number={},
        pages={736-740},
        keywords={Training;Conferences;Employment;Signal processing;Task analysis;Speech processing;Tuning;audio captioning;dataset;Clotho},
        doi={10.1109/ICASSP40776.2020.9052990}
    }
