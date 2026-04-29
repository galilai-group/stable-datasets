"""Video dataset example for stable-datasets."""

from stable_datasets import VideoDecodeConfig
from stable_datasets.video import SSv2


def main() -> None:
    ds = SSv2(
        split="train",
        storage_format="lance",
    )

    sample = ds[0]
    ref = sample["video"]

    print("Default video access")
    print("video field type:", type(ref).__name__)
    print("cached path:", ref.path)
    print("extension:", ref.extension)
    print("media_type:", ref.media_type)

    decoded = ds.set_video_decode(
        VideoDecodeConfig(
            num_frames=16,
            sampling="uniform",
            decoder="torchcodec",
            output="torch",
            layout="TCHW",
        )
    )

    decoded_sample = decoded[0]
    print("\nDecoded video access")
    print("video tensor shape:", tuple(decoded_sample["video"].shape))
    print("label:", decoded_sample["label"])


if __name__ == "__main__":
    main()
