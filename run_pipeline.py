"""
run_pipeline.py
---------------
ResNet50-CNN-LSTM Alzheimer's classifier — PyTorch + CUDA (RTX 4050)

Usage:
    python run_pipeline.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score

# ── config ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EPOCHS      = 50
BATCH_SIZE  = 32
IMG_SIZE    = 224
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model ────────────────────────────────────────────────────────────────────
class ResNet50_CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # freeze only layer1 and below, unfreeze layer2/3/4
        for name, p in base.named_parameters():
            if any(k in name for k in ["layer2", "layer3", "layer4"]):
                p.requires_grad = True
            else:
                p.requires_grad = False

        # backbone: everything except the final FC
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )  # output: (B, 2048, 4, 4) for 128x128 input

        self.cnn_refine = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout2d(0.3),
            nn.Conv2d(512,  128, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.2),
        )  # output: (B, 128, 4, 4)

        # (B, 128, 4, 4) -> (B, 4, 512) for LSTM
        self.lstm = nn.LSTM(input_size=128*7, hidden_size=256, num_layers=1,
                            batch_first=True, dropout=0.0)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)           # (B, 2048, H, W)
        x = self.cnn_refine(x)         # (B, 128, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)     # (B, H, W, C)
        x = x.reshape(B, H, W * C)    # (B, H, W*C) — H as time steps
        x, _ = self.lstm(x)            # (B, H, 256)
        x = x[:, -1, :]               # (B, 256)
        return self.classifier(x)


# ── data ─────────────────────────────────────────────────────────────────────
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(DATASET_DIR)
    n_val = int(0.2 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val],
                            generator=torch.Generator().manual_seed(42))

    # apply transforms separately
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf

    # weighted sampler — oversample minority classes
    targets = [full.targets[i] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(targets))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader, full.classes


# ── train ─────────────────────────────────────────────────────────────────────
def train():
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Dataset : {DATASET_DIR}\n")

    with tqdm(total=2, desc="Setup", unit="step", colour="blue") as bar:
        bar.set_description("Loading dataset")
        train_loader, val_loader, classes = get_loaders()
        bar.update(1)

        bar.set_description("Building model")
        model = ResNet50_CNN_LSTM(num_classes=len(classes)).to(DEVICE)
        bar.update(1)

    print(f"\nClasses : {classes}")
    print(f"Train   : {len(train_loader.dataset)} images")
    print(f"Val     : {len(val_loader.dataset)} images\n")

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        [1.0 / len(os.listdir(os.path.join(DATASET_DIR, c))) for c in classes],
        dtype=torch.float
    ).to(DEVICE))
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": LR * 0.1},   # 3e-6 for VGG layers
        {"params": head_params,     "lr": LR},          # 3e-5 for new layers
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    best_val_f1  = 0.0
    patience     = 12
    patience_ctr = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch", colour="yellow")

    for epoch in epoch_bar:
        # ── train phase ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Epoch {epoch:02d} train",
                                  leave=False, unit="batch", colour="green"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        # ── val phase ──
        # collect val preds for F1
        val_preds, val_true = [], []
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"  Epoch {epoch:02d} val  ",
                                      leave=False, unit="batch", colour="cyan"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out   = model(imgs)
                loss  = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        t_l   = t_loss / t_total
        v_l   = v_loss / v_total
        v_f1  = f1_score(val_true, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(t_l)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_l)
        history["val_acc"].append(v_acc)

        scheduler.step()

        epoch_bar.set_postfix({
            "loss": f"{t_l:.4f}", "acc": f"{t_acc:.4f}",
            "val_loss": f"{v_l:.4f}", "val_acc": f"{v_acc:.4f}", "val_f1": f"{v_f1:.4f}",
        })
        tqdm.write(
            f"\n{'─'*55}\n"
            f"  Epoch {epoch:3d}/{EPOCHS} complete\n"
            f"  Train  — loss: {t_l:.4f}  acc: {t_acc*100:.2f}%\n"
            f"  Val    — loss: {v_l:.4f}  acc: {v_acc*100:.2f}%  macro_f1: {v_f1:.4f}\n"
            f"{'─'*55}"
        )

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), "best_alzheimer_resnet50.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                tqdm.write(f"\n  [EARLY STOP] No improvement for {patience} epochs.")
                break

    print(f"\nBest val F1 (macro): {best_val_f1:.4f}")

    # ── plots ──
    with tqdm(total=1, desc="Saving plots", unit="file", colour="magenta") as bar:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_acc"], label="Train Acc")
        axes[0].plot(history["val_acc"],   label="Val Acc")
        axes[0].set_title("Accuracy"); axes[0].legend()
        axes[1].plot(history["train_loss"], label="Train Loss")
        axes[1].plot(history["val_loss"],   label="Val Loss")
        axes[1].set_title("Loss"); axes[1].legend()
        plt.tight_layout()
        plt.savefig("training_curves.png")
        bar.update(1)

    # ── classification report ──
    print("\nEvaluating on validation set...")
    model.load_state_dict(torch.load("best_alzheimer_resnet50.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Predicting", unit="batch", colour="cyan"):
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\n", report)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    with open("class_indices.json", "w") as f:
        json.dump({c: i for i, c in enumerate(classes)}, f)

    print("Done. Model saved -> best_alzheimer_model.pth")


if __name__ == "__main__":
    print("=" * 55)
    print("  OASIS Alzheimer's Classification — PyTorch")
    print("=" * 55 + "\n")
    train()
