import json

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import CLEVR


pytestmark = pytest.mark.large


def test_clevr_dataset():
    # CLEVR(split="train") automatically downloads and loads the dataset
    clevr_train = CLEVR(split="train")

    # Test 1: Check that the dataset has the expected number of training samples
    expected_num_train_samples = 70000
    assert len(clevr_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(clevr_train)}."
    )

    # Test 2: Check that each sample has the expected keys
    sample = clevr_train[0]
    expected_keys = {"image", "image_filename", "image_index", "scene_json", "questions_json"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and number of channels
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, f"Image should be HxWx3, got shape {image_np.shape}"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate image_filename
    image_filename = sample["image_filename"]
    assert isinstance(image_filename, str), f"image_filename should be a string, got {type(image_filename)}."
    assert image_filename.endswith(".png"), f"image_filename should end with .png, got {image_filename}."
    assert image_filename.startswith("CLEVR_train_"), (
        f"Train image_filename should start with 'CLEVR_train_', got {image_filename}."
    )

    # Test 5: Validate image_index type and range
    image_index = sample["image_index"]
    assert isinstance(image_index, int), f"image_index should be an integer, got {type(image_index)}."
    assert 0 <= image_index < 70000, f"image_index should be in [0, 70000), got {image_index}."

    # Test 6: Validate scene_json contains expected fields
    scene = json.loads(sample["scene_json"])
    assert isinstance(scene, dict), f"scene_json should decode to a dict, got {type(scene)}."
    assert "objects" in scene, "scene should have an 'objects' key."
    assert "relationships" in scene, "scene should have a 'relationships' key."
    assert isinstance(scene["objects"], list), "'objects' should be a list."

    # Test 7: Validate at least one object has the expected attribute keys
    if scene["objects"]:
        obj = scene["objects"][0]
        for attr in ("color", "material", "shape", "size", "3d_coords", "rotation"):
            assert attr in obj, f"Object should have '{attr}' attribute."

    # Test 8: Validate questions_json
    questions = json.loads(sample["questions_json"])
    assert isinstance(questions, list), f"questions_json should decode to a list, got {type(questions)}."
    if questions:
        q = questions[0]
        assert "question" in q, "question dict should have a 'question' key."
        assert "answer" in q, "Train questions should have an 'answer' key."

    # Test 9: Check the validation split
    clevr_val = CLEVR(split="validation")
    expected_num_val_samples = 15000
    assert len(clevr_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(clevr_val)}."
    )

    # Test 10: Validation split should also have scene and question data
    val_sample = clevr_val[0]
    val_scene = json.loads(val_sample["scene_json"])
    assert isinstance(val_scene, dict) and "objects" in val_scene, "Validation split should have scene annotations."

    # Test 11: Check the test split
    clevr_test = CLEVR(split="test")
    expected_num_test_samples = 15000
    assert len(clevr_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(clevr_test)}."
    )

    # Test 12: Test split should have empty scene data
    test_sample = clevr_test[0]
    test_scene = json.loads(test_sample["scene_json"])
    assert test_scene == {}, "Test split should have no scene annotations (empty dict)."

    # Test 13: Test split questions should not have 'answer' or 'program' keys
    test_questions = json.loads(test_sample["questions_json"])
    assert isinstance(test_questions, list), "test questions_json should be a list."
    if test_questions:
        tq = test_questions[0]
        assert "answer" not in tq, "Test split questions should not have an 'answer' key."
        assert "program" not in tq, "Test split questions should not have a 'program' key."

    print("All CLEVR dataset tests passed successfully!")


if __name__ == "__main__":
    test_clevr_dataset()
