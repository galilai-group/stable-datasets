import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_datasets.images import Chars74K


SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 64
NUM_CLASSES = 62


class CharsDataset(Dataset):
    def __init__(self, split):
        self.data = Chars74K(split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        image = sample["image"].convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

        pixels = np.asarray(image, dtype=np.float32).copy()
        pixels = torch.from_numpy(pixels).permute(2, 0, 1) / 255.0

        return pixels, int(sample["label"])


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), 100.0 * correct / len(loader.dataset)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_data = CharsDataset("train")
    test_data = CharsDataset("test")

    print("Device:", device)
    print("Train:", len(train_data))
    print("Test:", len(test_data))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(round(train_loss, 6))
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}")

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    results = {
        "dataset": "Chars74K EnglishHnd",
        "model": "SmallCNN",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "device": str(device),
        "split_method": "sample_id_modulo_5",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_losses": train_losses,
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 4),
    }

    output = Path("examples/results/chars74k_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    print("Results saved to:", output)


if __name__ == "__main__":
    main()
