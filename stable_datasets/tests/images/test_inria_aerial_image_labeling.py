import shutil
import warnings

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.inria_aerial_image_labeling import InriaAerialImageLabeling
from stable_datasets.utils import KAGGLE_CLI_SETUP_INSTRUCTIONS


@pytest.mark.large
def test_inria_aerial_image_labeling_dataset():
    if shutil.which("kaggle") is None:
        warnings.warn(
            "Kaggle CLI not found.\n" + KAGGLE_CLI_SETUP_INSTRUCTIONS,
            stacklevel=2,
        )
        pytest.skip("Kaggle CLI not found. See warning above for install and API key setup.")

    try:
        train_ds = InriaAerialImageLabeling(split="train")
    except RuntimeError as error:
        err = str(error)
        if "[kaggle:SIGKILL]" in err:
            warnings.warn(
                "Kaggle download was killed (SIGKILL), often due to low memory while fetching a huge archive.\n"
                + err,
                stacklevel=2,
            )
            pytest.skip(
                "Kaggle download SIGKILL (likely OOM). Pre-download with kaggle CLI in a shell, free RAM, then re-run."
            )
        if "Kaggle download failed" in err or "[kaggle:failed]" in err:
            # Typical when ~/.kaggle/kaggle.json or env vars are missing (CLI is installed).
            if "KeyError" in err and "username" in err:
                warnings.warn(
                    "Kaggle API credentials are not configured.\n" + KAGGLE_CLI_SETUP_INSTRUCTIONS,
                    stacklevel=2,
                )
                pytest.skip(
                    "Kaggle credentials missing (see warning above). "
                    "Configure ~/.kaggle/kaggle.json or export KAGGLE_USERNAME and KAGGLE_KEY, then re-run."
                )
            pytest.fail(
                "Kaggle download failed. Fix authentication or network, then retry.\n" + KAGGLE_CLI_SETUP_INSTRUCTIONS
            )
        raise

    assert len(train_ds) > 0, "Expected non-empty train split."

    sample = train_ds[0]
    expected_keys = {"image", "mask"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    image = sample["image"]
    assert isinstance(image, Image.Image), f"image should be PIL.Image, got {type(image)}."
    image_np = np.array(image)
    assert len(image_np.shape) == 3, f"image should be HxWxC, got {image_np.shape}"

    mask = sample["mask"]
    assert isinstance(mask, Image.Image), f"mask should be PIL.Image, got {type(mask)}."

    try:
        test_ds = InriaAerialImageLabeling(split="test")
    except ValueError as error:
        if "not found" in str(error):
            return
        raise
    if len(test_ds) == 0:
        return
    test_sample = test_ds[0]
    assert set(test_sample.keys()) == expected_keys
    assert test_sample["image"] is not None
    assert test_sample["mask"] is None
