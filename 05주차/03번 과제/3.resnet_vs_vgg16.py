"""
ResNet50 vs VGG16 성능 비교 실험 (전이 학습 기반)
Dataset : CIFAR-10 (cat=3, dog=5 → 이진 분류)  ← 02번 과제와 동일
공통 조건 : ImageNet 사전 학습 가중치, conv 동결, FC만 fine-tuning
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights, VGG16_Weights

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("../data")
RESULTS_DIR = Path("results")
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
TRAIN_N = 200
VAL_N = 50
TEST_N = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["cat", "dog"]
_CIFAR_CAT = 3
_CIFAR_DOG = 5


# ── Dataset (02번 과제와 동일) ────────────────────────────────────────────────
def get_transform(train: bool) -> transforms.Compose:
    aug = [transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(
        [
            transforms.Resize(224),
            *aug,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class CatDogCIFAR(Dataset):
    """CIFAR-10에서 cat(3)·dog(5)만 추출, 0=cat / 1=dog로 재매핑"""

    def __init__(self, root: Path, train: bool, transform):
        base = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        self.samples = [
            (img, 0 if lbl == _CIFAR_CAT else 1)
            for img, lbl in base
            if lbl in (_CIFAR_CAT, _CIFAR_DOG)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def balanced_subset(dataset: CatDogCIFAR, n: int, offset: int = 0) -> Subset:
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for i, (_, label) in enumerate(dataset.samples):
        by_class[label].append(i)
    selected = []
    for c in [0, 1]:
        selected.extend(by_class[c][offset : offset + n])
    return Subset(dataset, selected)


# ── Models ───────────────────────────────────────────────────────────────────
def build_resnet50() -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(2048, len(CLASSES))
    return model.to(DEVICE)


def build_vgg16() -> nn.Module:
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, len(CLASSES))
    return model.to(DEVICE)


# ── Train / Eval ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


# ── Experiment ───────────────────────────────────────────────────────────────
def run_experiment(
    name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}  |  device={DEVICE}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()

    _, pre_test_acc = evaluate(model, test_loader, criterion)
    print(f"  [Before Training] Test Accuracy : {pre_test_acc*100:.1f}%")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    history: dict = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    history["pre_test_acc"] = round(pre_test_acc, 6)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        for k, v in zip(history, [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(round(v, 6))
        print(
            f"  Epoch {epoch}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}"
        )

    elapsed = time.time() - t0
    _, test_acc = evaluate(model, test_loader, criterion)
    history["test_acc"] = round(test_acc, 6)
    history["elapsed"] = round(elapsed, 1)

    print(f"\n  ▶ [After Training] Test Accuracy : {test_acc*100:.1f}%")
    print(f"  ▶ Training Time                  : {elapsed:.1f}s")
    return history


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading CIFAR-10 (cat/dog subset)...")
    train_full = CatDogCIFAR(DATA_DIR, train=True, transform=get_transform(True))
    val_full = CatDogCIFAR(DATA_DIR, train=True, transform=get_transform(False))
    test_full = CatDogCIFAR(DATA_DIR, train=False, transform=get_transform(False))

    train_subset = balanced_subset(train_full, TRAIN_N, offset=0)
    val_subset = balanced_subset(val_full, VAL_N, offset=TRAIN_N)
    test_subset = balanced_subset(test_full, TEST_N, offset=0)

    print(
        f"  Train={len(train_subset)}  Val={len(val_subset)}  Test={len(test_subset)}"
    )
    print(f"  Device: {DEVICE}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=0)
    train_loader = DataLoader(train_subset, shuffle=True, **kw)
    val_loader = DataLoader(val_subset, shuffle=False, **kw)
    test_loader = DataLoader(test_subset, shuffle=False, **kw)

    results: dict = {}
    results["ResNet50"] = run_experiment(
        "ResNet50",
        build_resnet50(),
        train_loader,
        val_loader,
        test_loader,
    )
    results["VGG16"] = run_experiment(
        "VGG16",
        build_vgg16(),
        train_loader,
        val_loader,
        test_loader,
    )

    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── Summary ──
    r = results["ResNet50"]
    v = results["VGG16"]

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'':35s}  {'ResNet50':>9}  {'VGG16':>9}")
    print(
        f"  {'Test Acc (before training)':35s}  {r['pre_test_acc']*100:>8.1f}%  {v['pre_test_acc']*100:>8.1f}%"
    )
    print(
        f"  {'Test Acc (after training)':35s}  {r['test_acc']*100:>8.1f}%  {v['test_acc']*100:>8.1f}%"
    )
    print(
        f"  {'Final Val Accuracy':35s}  {r['val_acc'][-1]*100:>8.1f}%  {v['val_acc'][-1]*100:>8.1f}%"
    )
    print(f"  {'Training Time (s)':35s}  {r['elapsed']:>9.1f}  {v['elapsed']:>9.1f}")


if __name__ == "__main__":
    main()
