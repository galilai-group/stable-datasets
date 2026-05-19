"""Time-series / audio dataset example for stable-datasets."""

from stable_datasets.timeseries import AudioMNIST


def main() -> None:
    ds = AudioMNIST(split="train")
    sample = ds[0]

    print("Default format")
    print("keys:", sample.keys())
    print("label:", sample["label"])
    print("speaker_id:", sample["speaker_id"])
    print("channels:", len(sample["series"]))

    ds_np = ds.with_format("numpy")
    np_sample = ds_np[0]

    print("\nNumPy format")
    print("series shape:", np_sample["series"].shape)
    print("series dtype:", np_sample["series"].dtype)


if __name__ == "__main__":
    main()
