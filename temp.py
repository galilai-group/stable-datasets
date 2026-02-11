# !pip instasll torchvision

from stable_datasets.images.not_mnist import NotMNIST
from torchvision import transforms

ds = NotMNIST(split="train")
ds_torch = ds.with_format("torch")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

sample = ds[0]
print(sample.keys())  # {"image", "label"}

# Access the image and label
image = sample["image"]  # PIL.Image.Image
label = sample["label"]  # int (0-9)

tensor = transform(image)
print(f"Tensor shape: {tensor.shape}")  # torch.Size([1, 28, 28])
print(f"Label: {label} -> Letter: {chr(ord('A') + label)}")
