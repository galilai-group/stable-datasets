"""Train ResNet-18 on a selectable CIFAR-N label field.

Examples:
    python -m benchmarks.run_cifar_n --dataset cifar10n --label aggregate
    python -m benchmarks.run_cifar_n --dataset cifar10n --label random_2 --epochs 25
    python -m benchmarks.run_cifar_n --dataset cifar100n --label noisy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from stable_datasets.images import CIFAR10N, CIFAR100N
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATASETS = ["cifar10n", "cifar100n"]

LABEL_FIELDS = {
    "cifar10n": {
        "clean": "label",
        "aggregate": "aggre_label",
        "worst": "worse_label",
        "random_1": "random_label1",
        "random_2": "random_label2",
        "random_3": "random_label3",
    },
    "cifar100n": {"clean": "label", "noisy": "noisy_label"},
}

CIFAR_STATS = {
    "cifar10n": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100n": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class LabelledCIFAR(Dataset):
    """Expose each source image paired with exactly one chosen label."""

    def __init__(self, dataset, label_field: str, image_transform):
        self.dataset = dataset
        self.label_field = label_field
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        label = sample.get(self.label_field)
        if label is None:
            raise ValueError(f"Missing label {self.label_field!r} at dataset index {index}.")
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.image_transform(image), int(label)


def _make_model(num_classes: int) -> nn.Module:
    """Create ResNet-18 with a CIFAR-sized stem and output layer."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / count, correct / count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10n")
    parser.add_argument("--label", help="Training target; choices depend on --dataset.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional stable-datasets cache root.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    choices = LABEL_FIELDS[args.dataset]
    if args.label is None:
        args.label = "aggregate" if args.dataset == "cifar10n" else "noisy"
    if args.label not in choices:
        parser.error(f"--label for {args.dataset} must be one of: {', '.join(choices)}")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("--epochs and --batch-size must be positive; --num-workers cannot be negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    builder_class = {"cifar10n": CIFAR10N, "cifar100n": CIFAR100N}[args.dataset]
    builder_kwargs = {}
    if args.data_dir is not None:
        builder_kwargs = {
            "download_dir": str(args.data_dir / "downloads"),
            "processed_cache_dir": str(args.data_dir / "processed"),
        }

    logging.info(
        "Loading %s; train target=%s (%s); test target=clean label",
        args.dataset,
        args.label,
        choices[args.label],
    )
    train_source = builder_class(split="train", **builder_kwargs)
    test_source = builder_class(split="test", **builder_kwargs)
    mean, std = CIFAR_STATS[args.dataset]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_dataset = LabelledCIFAR(train_source, choices[args.label], train_transform)
    test_dataset = LabelledCIFAR(test_source, "label", test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = _make_model(10 if args.dataset == "cifar10n" else 100).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Training ResNet-18 on %s", device)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        logging.info(
            "epoch %d/%d train_loss=%.4f train_accuracy=%.2f%%",
            epoch,
            args.epochs,
            train_loss,
            100 * train_accuracy,
        )

    test_loss, test_accuracy = _run_epoch(model, test_loader, loss_fn, device, optimizer=None)
    logging.info(
        "test_clean_loss=%.4f test_clean_accuracy=%.2f%% (%s trained on %s labels)",
        test_loss,
        100 * test_accuracy,
        args.dataset,
        args.label,
    )
    return test_accuracy


if __name__ == "__main__":
    main()
