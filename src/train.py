# src/train.py
import os
import random
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import MultiHeadDataset
from model import DualHeadClassifier


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-Head Multi-Label Trainer")

    # paths
    parser.add_argument("--data-dir", type=str, default="data/cropped_maps", help="image directory")
    parser.add_argument("--label-dir", type=str, default="data/labels", help="label directory")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="checkpoint directory")

    # train config
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="validation ratio in [0,1]")

    # data/loader
    parser.add_argument("--img-size", type=int, default=224, help="input image size (square)")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # model / metrics
    parser.add_argument("--num-classes", type=int, default=4, help="number of classes")
    parser.add_argument("--k-top", type=int, default=3, help="top-K for HeadB validation")

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])

    return parser.parse_args()


def build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    tfm = build_transforms(args.img_size)
    full_ds = MultiHeadDataset(args.data_dir, args.label_dir, transform=tfm)

    val_size = int(len(full_ds) * args.val_ratio)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f" Dataset split: train={len(train_ds)} | val={len(val_ds)}")
    return train_dl, val_dl


def train_one_epoch(model: nn.Module,
                    train_dl: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion_a: nn.Module,
                    criterion_b: nn.Module,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for imgs, label_a, label_b in train_dl:
        imgs = imgs.to(device)
        label_a = label_a.to(device)
        label_b = label_b.to(device)

        out_a, out_b_logits = model(imgs)
        loss_a = criterion_a(out_a, label_a)
        loss_b = criterion_b(out_b_logits, label_b)
        loss = loss_a + loss_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(train_dl))


@torch.no_grad()
def validate_one_epoch(model: nn.Module,
                       val_dl: DataLoader,
                       criterion_a: nn.Module,
                       criterion_b: nn.Module,
                       k_top: int,
                       device: torch.device) -> Dict[str, float]:
    model.eval()
    val_loss = 0.0

    # metrics
    correct_a, total_a = 0, 0
    correct_b_thr, correct_b_topk, total_b = 0, 0, 0

    for imgs, label_a, label_b in val_dl:
        imgs = imgs.to(device)
        label_a = label_a.to(device)
        label_b = label_b.to(device)

        out_a, out_b_logits = model(imgs)

        # --- safety guard for batch mismatch ---
        if out_b_logits.size(0) != label_b.size(0):
            mb = min(out_b_logits.size(0), label_b.size(0))
            out_a, out_b_logits = out_a[:mb], out_b_logits[:mb]
            label_a, label_b = label_a[:mb], label_b[:mb]

        loss_a = criterion_a(out_a, label_a)
        loss_b = criterion_b(out_b_logits, label_b)
        val_loss += (loss_a + loss_b).item()

        # Head A acc
        preds_a = torch.argmax(out_a, dim=1)
        correct_a += (preds_a == label_a).sum().item()
        total_a += label_a.size(0)

        # Head B metrics
        probs_b = torch.sigmoid(out_b_logits)  # (B,C)

        # (1) threshold accuracy (0.5)
        preds_b_thr = (probs_b > 0.5).float()
        correct_b_thr += (preds_b_thr == label_b).all(dim=1).sum().item()

        # (2) top-K accuracy
        topk_idx = probs_b.topk(k_top, dim=1).indices
        preds_b_topk = torch.zeros_like(probs_b)
        preds_b_topk.scatter_(1, topk_idx, 1.0)
        correct_b_topk += (preds_b_topk == label_b).all(dim=1).sum().item()

        total_b += label_b.size(0)

    val_loss /= max(1, len(val_dl))
    val_acc_a = (correct_a / total_a * 100.0) if total_a else 0.0
    val_acc_b_thr = (correct_b_thr / total_b * 100.0) if total_b else 0.0
    val_acc_b_topk = (correct_b_topk / total_b * 100.0) if total_b else 0.0

    return {
        "val_loss": val_loss,
        "val_acc_a": val_acc_a,
        "val_acc_b_thr": val_acc_b_thr,
        "val_acc_b_topk": val_acc_b_topk
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device(args.device)
    model = DualHeadClassifier(num_classes=args.num_classes).to(device)

    criterion_a = nn.CrossEntropyLoss()
    criterion_b = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dl, val_dl = build_dataloaders(args)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion_a, criterion_b, device)
        metrics = validate_one_epoch(model, val_dl, criterion_a, criterion_b, args.k_top, device)

        print(f"[Epoch {epoch+1:02d}] "
              f"TrainLoss={train_loss:.4f} | "
              f"ValLoss={metrics['val_loss']:.4f} | "
              f"ValAccA={metrics['val_acc_a']:.2f}% | "
              f"ValAccB(thr@0.5)={metrics['val_acc_b_thr']:.2f}% | "
              f"ValAccB(top{args.k_top})={metrics['val_acc_b_topk']:.2f}%")

        # save best
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch + 1
            save_path = os.path.join(args.ckpt_dir, f"best_model_epoch{best_epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best model updated and saved at {save_path}")

    print(f"\n Best Epoch = {best_epoch} (ValLoss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
