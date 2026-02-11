from stable_datasets.images.tiny_imagenet_c import TinyImagenetC

print("Loading Tiny ImageNet-C dataset...")

tiny_c_test = TinyImagenetC(split="test")
tiny_c_all = TinyImagenetC(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {tiny_c_test.info.homepage}")
print(f"  - Description: {tiny_c_test.info.description}")
print(f"  - Citation:\n{tiny_c_test.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Test samples: {len(tiny_c_test)}")
print(f"  - Total splits: {len(tiny_c_all)}")
print(f"  - Total samples (test): {len(tiny_c_test)}")

sample = tiny_c_test[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Image type: {type(sample['image'])}")
print(f"  - Image size: {sample['image'].size}")
print(f"  - Label (string): {sample.get('label')}")
print(f"  - Corruption name: {sample.get('corruption_name')}")
print(f"  - Corruption level: {sample.get('corruption_level')}")

print("\nTiny ImageNet-C dataset loaded successfully!")