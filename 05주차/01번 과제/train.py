"""Fine-tuning script for dog/cat classification on a custom dataset.

Expected dataset directory layout:
    data/
        train/
            dog/  *.jpg ...
            cat/  *.jpg ...
        val/
            dog/  *.jpg ...
            cat/  *.jpg ...
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier import IMAGENET_TRANSFORM, _build_finetuned_resnet

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train(
    data_dir: str | Path,
    output_path: str | Path = "dogcat_resnet50.pth",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    data_dir = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=IMAGENET_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"Classes: {train_ds.classes}")

    model = _build_finetuned_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_acc, val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total, loss_sum / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet50 for dog/cat classification"
    )
    parser.add_argument(
        "--data", required=True, help="Path to dataset root (train/val subdirs)"
    )
    parser.add_argument(
        "--output", default="dogcat_resnet50.pth", help="Output weights path"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data, args.output, args.epochs, args.batch_size, args.lr)
