from pathlib import Path

import numpy as np

from stable_datasets.schema import DatasetInfo, DatasetSource, DownloadInfo, Features, Sequence, Value, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class SeizuresNeonatal(BaseDatasetBuilder):
    """Neonatal EEG recordings with expert seizure annotations."""

    VERSION = Version("1.0.0")
    SOURCE = DatasetSource(
        homepage="https://zenodo.org/records/2547147",
        assets={
            "annotations": DownloadInfo(
                url="https://zenodo.org/record/2547147/files/annotations_2017.mat?download=1",
                fallbacks=["https://zenodo.org/records/2547147/files/annotations_2017.mat"],
                filename="annotations_2017.mat",
            ),
            **{
                f"eeg{i}": DownloadInfo(
                    url=f"https://zenodo.org/record/2547147/files/eeg{i}.edf?download=1",
                    fallbacks=[f"https://zenodo.org/records/2547147/files/eeg{i}.edf"],
                    filename=f"eeg{i}.edf",
                )
                for i in range(1, 80)
            },
        },
        citation="See dataset homepage.",
    )

    def _info(self):
        return DatasetInfo(
            description="Multichannel neonatal EEG recordings with expert seizure annotations.",
            features=Features(
                {
                    "series": Sequence(Sequence(Value("float32"))),
                    "annotations": Sequence(Sequence(Value("int32"))),
                    "subject_id": Value("int32"),
                    "filename": Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        asset_names = ["annotations", *[f"eeg{i}" for i in range(1, 80)]]
        local_paths = bulk_download(
            [self._normalize_download_info(source["assets"][name], asset_name=name) for name in asset_names],
            dest_folder=self._raw_download_dir,
        )
        annotations_path = local_paths[0]
        eeg_paths = local_paths[1:]
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"annotations_path": annotations_path, "eeg_paths": eeg_paths, "split": "train"},
            )
        ]

    def _candidate_splits(self) -> list:
        return [Split.TRAIN]

    def _generate_examples(self, annotations_path, eeg_paths, split):
        del split
        import mne
        from scipy.io import loadmat

        annotations = loadmat(annotations_path)["annotat_new"][0]
        eeg_by_id = {int(Path(path).stem.replace("eeg", "")): path for path in eeg_paths}
        for subject_id in range(1, 80):
            eeg_path = eeg_by_id[subject_id]
            raw = mne.io.read_raw_edf(str(eeg_path), preload=True, verbose="ERROR")
            series = raw.get_data().T.astype("float32")
            annotation = np.asarray(annotations[subject_id - 1]).astype("int32")
            if annotation.ndim == 1:
                annotation = annotation[:, None]
            yield (
                subject_id,
                {
                    "series": series,
                    "annotations": annotation.tolist(),
                    "subject_id": subject_id,
                    "filename": Path(eeg_path).name,
                },
            )
