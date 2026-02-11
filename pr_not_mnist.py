from stable_datasets.images.not_mnist import NotMNIST

print("Loading NotMNIST dataset...")
notmnist_train = NotMNIST(split="train")
notmnist_test = NotMNIST(split="test")
notmnist_all = NotMNIST(split=None)

print(f"\nDataset Metadata:")
print(f"  - Homepage: {notmnist_train.info.homepage}")
print(f"  - Description: {notmnist_train.info.description}")
print(f"  - Citation:\n{notmnist_train.info.citation}")

print(f"\nDataset Statistics:")
print(f"  - Train samples: {len(notmnist_train)}")
print(f"  - Test samples: {len(notmnist_test)}")
print(f"  - Total splits: {len(notmnist_all)}")
print(f"  - Total samples: {len(notmnist_all['train']) + len(notmnist_all['test'])}")
print(f"  - Number of classes: {notmnist_train.features['label'].num_classes}")

sample = notmnist_train[0]
print(f"\nSample Information:")
print(f"  - Keys: {list(sample.keys())}")
print(f"  - Image type: {type(sample['image'])}")
print(f"  - Image size: {sample['image'].size}")
print(f"  - Label (int): {sample['label']}")
print(f"  - Label (string): {notmnist_train.features['label'].int2str(sample['label'])}")

print(f"\nAll classes (A-J):")
for i in range(10):
    print(f"  {i}: {notmnist_train.features['label'].names[i]}")

print("\nNotMNIST dataset loaded successfully!")
