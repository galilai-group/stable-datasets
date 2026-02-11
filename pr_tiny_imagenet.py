from stable_datasets.images.tiny_imagenet import TinyImagenet

print("Loading Tiny ImageNet dataset...")

tiny_train = TinyImagenet(split="train")
tiny_val = TinyImagenet(split="validation")
tiny_all = TinyImagenet(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {tiny_train.info.homepage}")
print(f"  - Description: {tiny_train.info.description}")
print(f"  - Citation:\n{tiny_train.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Train samples: {len(tiny_train)}")
print(f"  - Validation samples: {len(tiny_val)}")
print(f"  - Total splits: {len(tiny_all)}")
print(f"  - Total samples (train+val): {len(tiny_train) + len(tiny_val)}")
print(f"  - Number of classes: {tiny_train.features['label'].num_classes}")

sample = tiny_train[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Image type: {type(sample['image'])}")
print(f"  - Image size: {sample['image'].size}")
print(f"  - Label (int): {sample['label']}")
# If label supports int->str mapping, show string form
try:
    print(f"  - Label (string): {tiny_train.features['label'].int2str(sample['label'])}")
except Exception:
    pass

print("\nTiny ImageNet dataset loaded successfully!")