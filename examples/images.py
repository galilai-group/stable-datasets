"""Image dataset example for stable-datasets."""

from stable_datasets.images import CIFAR10


def main() -> None:
    ds = CIFAR10(split="train")
    sample = ds[0]

    print("Default format")
    print("keys:", sample.keys())
    print("image type:", type(sample["image"]).__name__)
    print("label:", sample["label"])

    ds_torch = ds.with_format("torch")
    torch_sample = ds_torch[0]

    print("\nTorch format")
    print("image shape:", tuple(torch_sample["image"].shape))
    print("image dtype:", torch_sample["image"].dtype)


if __name__ == "__main__":
    main()
