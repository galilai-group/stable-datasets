from stable_datasets.images.resisc45 import RESISC45

print("Loading RESISC45 dataset...")
resisc45_train = RESISC45(split="train")
resisc45_all = RESISC45(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {resisc45_train.info.homepage}")
print(f"  - Description: {resisc45_train.info.description}")
print(f"  - Citation:\n{resisc45_train.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Train samples: {len(resisc45_train)}")
print(f"  - Total splits: {len(resisc45_all)}")
print(f"  - Number of classes: {resisc45_train.features['label'].num_classes}")

sample = resisc45_train[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Image type: {type(sample['image'])}")
print(f"  - Image size: {sample['image'].size}")
print(f"  - Label (int): {sample['label']}")
print(f"  - Label (string): {resisc45_train.features['label'].int2str(sample['label'])}")

print(f"\nFirst 10 class names:")
for i in range(10):
    print(f"  {i}: {resisc45_train.features['label'].names[i]}")

print("\nRESISC45 dataset loaded successfully!")
