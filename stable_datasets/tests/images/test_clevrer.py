import json

from stable_datasets.images.clevrer import CLEVRER


def test_clevrer_dataset():
    # CLEVRER(split="train") automatically downloads and loads the dataset
    clevrer_train = CLEVRER(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 10000
    assert len(clevrer_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(clevrer_train)}."
    )

    # Test 2: Check that each sample has the expected keys
    sample = clevrer_train[0]
    expected_keys = {"video", "scene_index", "video_filename", "questions_json", "annotations_json"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate scene_index type
    scene_index = sample["scene_index"]
    assert isinstance(scene_index, int), f"scene_index should be an integer, got {type(scene_index)}."

    # Test 4: Validate video_filename type
    video_filename = sample["video_filename"]
    assert isinstance(video_filename, str), f"video_filename should be a string, got {type(video_filename)}."
    assert video_filename.endswith(".mp4"), f"video_filename should end with .mp4, got {video_filename}."

    # Test 5: Validate questions_json is valid JSON
    questions = json.loads(sample["questions_json"])
    assert isinstance(questions, list), f"questions should be a list, got {type(questions)}."
    if len(questions) > 0:
        q = questions[0]
        assert "question_id" in q, "question should have 'question_id'"
        assert "question" in q, "question should have 'question'"
        assert "question_type" in q, "question should have 'question_type'"

    # Test 6: Validate annotations_json is valid JSON
    annotations_json = sample["annotations_json"]
    annotations = json.loads(annotations_json)
    assert isinstance(annotations, dict), "annotations should be a dict"

    # Test 7: Check the validation split
    clevrer_val = CLEVRER(split="validation")
    expected_num_val_samples = 5000
    assert len(clevrer_val) == expected_num_val_samples, (
        f"Expected {expected_num_val_samples} validation samples, got {len(clevrer_val)}."
    )

    # Test 8: Check the test split
    clevrer_test = CLEVRER(split="test")
    expected_num_test_samples = 5000
    assert len(clevrer_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(clevrer_test)}."
    )

    print("All CLEVRER dataset tests passed successfully!")


if __name__ == "__main__":
    test_clevrer_dataset()
