"""Load UCI HAR windows and MHEALTH recordings; no training or windowing."""

import numpy as np

from stable_datasets.timeseries import MHEALTH, UCIHAR


def main():
    har = UCIHAR()
    for split in ("train", "test"):
        sample = har[split][0]
        series = np.asarray(sample["series"], dtype=np.float32)
        print(f"UCI HAR {split}: {len(har[split])} windows; sample shape {series.shape}")
        print(f"  label={sample['label']}, subject_id={sample['subject_id']}")

    recordings = MHEALTH(split="all")
    print(f"MHEALTH: {len(recordings)} complete subject recordings (no evaluation split)")
    for record in recordings:
        series = np.asarray(record["series"], dtype=np.float32)
        labels = np.asarray(record["label"], dtype=np.int64)
        print(f"  subject={record['subject_id']}, shape={series.shape}, null samples={np.count_nonzero(labels == 0)}")


if __name__ == "__main__":
    main()
