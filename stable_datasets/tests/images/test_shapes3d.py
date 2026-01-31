import numpy as np
from PIL import Image

from stable_datasets.images.shapes3d import Shapes3D


def test_shapes3d_dataset():
    shapes3d_train = Shapes3D(split="train")

    expected_num_train_samples = 10 * 10 * 10 * 8 * 4 * 15
    assert len(shapes3d_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(shapes3d_train)}."
    )

    sample = shapes3d_train[0]
    expected_keys = {
        "image",
        "label",
        "label_index",
        "floor",
        "wall",
        "object",
        "scale",
        "shape",
        "orientation",
        "floor_idx",
        "wall_idx",
        "object_idx",
        "scale_idx",
        "shape_idx",
        "orientation_idx",
    }
    assert sorted(set(sample.keys())) == sorted(set(expected_keys)), (
        f"Expected keys {expected_keys}, got {set(sample.keys())}."
    )

    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL.Image.Image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Shapes3D images should be HxWx3, got shape {image_np.shape}."
    assert image_np.dtype == np.uint8, f"Image dtype should be uint8, got {image_np.dtype}."
    assert image_np.shape == (64, 64, 3), f"Image should have shape (64, 64, 3), got {image_np.shape}."

    label = sample["label"]
    label_index = sample["label_index"]
    assert isinstance(label, list), f"Label should be list, got {type(label)}."
    assert isinstance(label_index, list), f"Label index should be list, got {type(label_index)}."
    assert len(label) == 6, f"Label should have 6 elements, got {len(label)}."
    assert len(label_index) == 6, f"Label index should have 6 elements, got {len(label_index)}."

    floor = sample["floor"]
    wall = sample["wall"]
    obj = sample["object"]
    scale = sample["scale"]
    shape = sample["shape"]
    orientation = sample["orientation"]

    floor_idx = sample["floor_idx"]
    wall_idx = sample["wall_idx"]
    object_idx = sample["object_idx"]
    scale_idx = sample["scale_idx"]
    shape_idx = sample["shape_idx"]
    orientation_idx = sample["orientation_idx"]

    assert isinstance(floor, float | np.floating), f"floor should be float, got {type(floor)}."
    assert isinstance(wall, float | np.floating), f"wall should be float, got {type(wall)}."
    assert isinstance(obj, float | np.floating), f"object should be float, got {type(obj)}."
    assert isinstance(scale, float | np.floating), f"scale should be float, got {type(scale)}."
    assert isinstance(shape, float | np.floating), f"shape should be float, got {type(shape)}."
    assert isinstance(orientation, float | np.floating), f"orientation should be float, got {type(orientation)}."

    assert isinstance(floor_idx, int | np.integer), f"floor_idx should be int, got {type(floor_idx)}."
    assert isinstance(wall_idx, int | np.integer), f"wall_idx should be int, got {type(wall_idx)}."
    assert isinstance(object_idx, int | np.integer), f"object_idx should be int, got {type(object_idx)}."
    assert isinstance(scale_idx, int | np.integer), f"scale_idx should be int, got {type(scale_idx)}."
    assert isinstance(shape_idx, int | np.integer), f"shape_idx should be int, got {type(shape_idx)}."
    assert isinstance(orientation_idx, int | np.integer), (
        f"orientation_idx should be int, got {type(orientation_idx)}."
    )

    assert 0 <= floor_idx < 10, f"floor_idx should be in range [0, 9], got {floor_idx}."
    assert 0 <= wall_idx < 10, f"wall_idx should be in range [0, 9], got {wall_idx}."
    assert 0 <= object_idx < 10, f"object_idx should be in range [0, 9], got {object_idx}."
    assert 0 <= scale_idx < 8, f"scale_idx should be in range [0, 7], got {scale_idx}."
    assert 0 <= shape_idx < 4, f"shape_idx should be in range [0, 3], got {shape_idx}."
    assert 0 <= orientation_idx < 15, f"orientation_idx should be in range [0, 14], got {orientation_idx}."

    assert label[0] == floor, f"label[0] should equal floor, got {label[0]} vs {floor}."
    assert label[1] == wall, f"label[1] should equal wall, got {label[1]} vs {wall}."
    assert label[2] == obj, f"label[2] should equal object, got {label[2]} vs {obj}."
    assert label[3] == scale, f"label[3] should equal scale, got {label[3]} vs {scale}."
    assert label[4] == shape, f"label[4] should equal shape, got {label[4]} vs {shape}."
    assert label[5] == orientation, f"label[5] should equal orientation, got {label[5]} vs {orientation}."

    assert label_index[0] == floor_idx, f"label_index[0] should equal floor_idx, got {label_index[0]} vs {floor_idx}."
    assert label_index[1] == wall_idx, f"label_index[1] should equal wall_idx, got {label_index[1]} vs {wall_idx}."
    assert label_index[2] == object_idx, (
        f"label_index[2] should equal object_idx, got {label_index[2]} vs {object_idx}."
    )
    assert label_index[3] == scale_idx, f"label_index[3] should equal scale_idx, got {label_index[3]} vs {scale_idx}."
    assert label_index[4] == shape_idx, f"label_index[4] should equal shape_idx, got {label_index[4]} vs {shape_idx}."
    assert label_index[5] == orientation_idx, (
        f"label_index[5] should equal orientation_idx, got {label_index[5]} vs {orientation_idx}."
    )

    assert sample["floor"] == label[0], "floor should match label[0]."
    assert sample["wall"] == label[1], "wall should match label[1]."
    assert sample["object"] == label[2], "object should match label[2]."
    assert sample["scale"] == label[3], "scale should match label[3]."
    assert sample["shape"] == label[4], "shape should match label[4]."
    assert sample["orientation"] == label[5], "orientation should match label[5]."

    print("All Shapes3D dataset tests passed successfully!")
