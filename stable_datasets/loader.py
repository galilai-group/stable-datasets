import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch as ch


def _mapit(batch: tuple) -> list:
    with ch.no_grad():
        return [
            b.view(-1, *b.shape[2:]).to(device="cuda", non_blocking=True) for b in batch
        ]


class DatasetWithIndices:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class FastLoader:
    def __init__(self, dataset):
        self.length = len(dataset)
        self.dataset = iter(dataset)

        self.current = _mapit(next(self.dataset))
        self.ahead = _mapit(next(self.dataset))
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        current = self.current
        if self.count < len(self.dataset) - 2:
            self.current, self.ahead = self.ahead, _mapit(next(self.dataset))

        if self.count < len(self.dataset):
            self.count += 1
            return current
        raise StopIteration

    def __len__(self):
        return self.length


def dataset_to_h5(dataset, h5file, num_workers=16, chunk_size=1024):

    nfiles = len(dataset)

    loader = DataLoader(
        DatasetWithIndices(dataset),
        batch_size=chunk_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    with h5py.File(h5file, "w") as h5f:
        h5f.create_dataset("chunk_size", data=[chunk_size])

        label_ds = h5f.create_dataset("labels", shape=(nfiles,), dtype=int)
        indices_ds = h5f.create_dataset("indices", shape=(nfiles,), dtype=int)
        n = len(loader)
        for i, (x, y, indices) in tqdm(enumerate(loader), total=n, desc="converting"):
            h5f.create_dataset(f"images_{i}", data=x.numpy())
            label_ds[i * chunk_size : i * chunk_size + len(y)] = y
            indices_ds[i * chunk_size : i * chunk_size + len(y)] = indices


class H5Dataset(Dataset):
    def __init__(
        self,
        hdf5file,
        imgs_key="images",
        labels_key="labels",
        transform=None,
        device="cpu",
        chunkit=1,
        shuffle=True,
    ):
        self.shuffle = shuffle
        self.chunkit = chunkit
        self.hdf5file = hdf5file
        self.device = device
        self.imgs_key = imgs_key
        self.labels_key = labels_key
        self.transform = transform

        with h5py.File(self.hdf5file, "r") as db:
            self.lens = len(db[labels_key]) // chunkit
            self.datasets = [i for i in db.keys() if imgs_key in i]
            self.datasets.sort()
            self.chunk_size = db["chunk_size"][0]

        if not self.shuffle:
            assert chunkit == 1

    def __len__(self):
        return self.lens

    @ch.no_grad()
    def __getitem__(self, idx):
        idx *= self.chunkit
        chunk = idx // self.chunk_size
        index = idx - chunk * self.chunk_size
        with h5py.File(self.hdf5file, "r") as db:

            # unshuffle if needed
            if not self.shuffle:
                idx = db[f"indices"][idx]

            # we take care of chunking options loading
            if self.chunkit > 1:
                chunk_size = len(db[f"{self.imgs_key}_{chunk}"])
                indices = np.random.choice(
                    range(1, chunk_size), size=self.chunkit - 1, replace=False
                )
                indices = np.concatenate([[index], index + indices]) % chunk_size
                indices = np.sort(indices)
            else:
                # figure out the position w.r.t. chunks
                indices = np.array([index])

            # now proceed to load the data
            label = db[self.labels_key][indices + chunk * self.chunk_size]
            image = db[f"{self.imgs_key}_{chunk}"][indices, :, :, :]
            label = ch.from_numpy(label)
            image = ch.from_numpy(image)

            # apply the user-defined augmentation
            if self.transform:
                image = self.transform(image)
        return image, label
