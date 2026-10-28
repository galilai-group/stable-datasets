import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_datasets.images import SignLanguageMNIST


SEED = 42
EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SignDataset(Dataset):
    def __init__(self, split):
        self.data = SignLanguageMNIST(split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image = np.asarray(sample["image"].convert("L"), dtype=np.float32).copy()

        image = torch.from_numpy(image).unsqueeze(0) / 255.0
        label = int(sample["label"])

        return image, label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100 * correct / total


def main():
    set_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Device:", device)
    print("Loading datasets...")

    train_data = SignDataset("train")
    test_data = SignDataset("test")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:", len(train_data))
    print("Test samples:", len(test_data))

    model = SmallCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(round(train_loss, 6))

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}")

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    results = {
        "dataset": "Sign Language MNIST",
        "model": "SmallCNN",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "device": str(device),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_losses": train_losses,
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 4),
    }

    output = Path("examples/results/sign_language_mnist_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Results saved to:", output)


if __name__ == "__main__":
    main()
