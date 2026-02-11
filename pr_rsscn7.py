from stable_datasets.images.rsscn7 import RSSCN7

print("Loading RSSCN7 dataset...")
rsscn7_train = RSSCN7(split="train")
rsscn7_all = RSSCN7(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {rsscn7_train.info.homepage}")
print(f"  - Description: {rsscn7_train.info.description}")
print(f"  - Citation:\n{rsscn7_train.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Train samples: {len(rsscn7_train)}")
print(f"  - Total splits: {len(rsscn7_all)}")
print(f"  - Number of classes: {rsscn7_train.features['label'].num_classes}")

sample = rsscn7_train[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Image type: {type(sample['image'])}")
print(f"  - Image size: {sample['image'].size}")
print(f"  - Label (int): {sample['label']}")
print(f"  - Label (string): {rsscn7_train.features['label'].int2str(sample['label'])}")

print(f"\nAll class names:")
for i in range(7):
    print(f"  {i}: {rsscn7_train.features['label'].names[i]}")

print("\nRSSCN7 dataset loaded successfully!")
