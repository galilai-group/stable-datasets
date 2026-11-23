# Unreleased

- caches are Lance datasets only: `storage_format=`, `STORAGE_FORMAT`, the Arrow IPC writer and `ArrowBackend` are
  removed (passing `storage_format=` raises a `TypeError`); `pylance` is a core dependency; in-memory datasets use
  `TableBackend`; `map` and `flatten_indices` write Lance caches. Existing Lance caches keep their paths; Arrow caches
  are rebuilt as Lance on first load

# V0.0.0

- file locking for concurrent downloads
- download recovery from partially downloaded files
- datasets:
  - CIFAR10/CIFAR10C
  - CIFAR100/CIFAR100C
  - KMNIST
  - MedMNISTv2
  - FashionMNIST
  - Food101
  - FGVCAircraft
  - TinyImagenet/TinyImagenetC
