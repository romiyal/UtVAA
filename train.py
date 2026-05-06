"""
Training entry point for UltraLightBlockNet on CIFAR-100.

Usage:
    python train.py --config configs/cifar100.yaml --data-dir /path/to/cifar100
    python train.py --config configs/cifar100.yaml --variant tiny --epochs 300
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import v2

from utvaa.data import print_classwise_counts
from utvaa.engine import evaluate_model, train_one_epoch
from utvaa.models import UltraLightBlockNet_L1
from utvaa.utils import (
    plot_loss_landscape_diagram,
    plot_loss_landscape_with_library,
    plot_metrics,
    plot_tsne,
    save_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train UltraLightBlockNet on CIFAR-100")
    p.add_argument("--config",     default="configs/cifar100.yaml", help="YAML config path")
    p.add_argument("--data-dir",   default=None,   help="Override data.dir in config")
    p.add_argument("--output-dir", default=None,   help="Override output.dir in config")
    p.add_argument("--variant",    default=None,   choices=["tiny", "medium", "large", "xlarge"])
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--device",     default=None,   help="e.g. cuda:0, cuda:1, cpu")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args) -> dict:
    if args.data_dir:
        cfg["data"]["dir"] = args.data_dir
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.variant:
        cfg["model"]["variant"] = args.variant
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.device:
        cfg["device"] = args.device
    return cfg


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD  = [0.2673, 0.2564, 0.2762]


def build_train_transform(cfg: dict):
    aug = cfg["augmentation"]
    t = cfg["training"]
    sz = cfg["model"]["image_size"]
    return transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.RandomCrop(sz, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=aug["color_jitter"],
            contrast=aug["color_jitter"],
            saturation=aug["color_jitter"],
            hue=0.1,
        ),
        transforms.RandAugment(num_ops=aug["rand_augment_ops"], magnitude=aug["rand_augment_magnitude"]),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(
            p=aug["random_erasing_prob"],
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])


def build_val_transform(cfg: dict):
    sz = cfg["model"]["image_size"]
    return transforms.Compose([
        transforms.Resize(sz, interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def make_collate_fn(num_classes: int, mixup_alpha: float, cutmix_alpha: float):
    def collate_fn(batch):
        cutmix = v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha)
        mixup  = v2.MixUp(num_classes=num_classes,  alpha=mixup_alpha)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[0.5, 0.5])
        return cutmix_or_mixup(*default_collate(batch))
    return collate_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = merge_args(load_config(args.config), args)

    tr_cfg  = cfg["training"]
    aug_cfg = cfg["augmentation"]
    dat_cfg = cfg["data"]
    out_cfg = cfg["output"]
    mdl_cfg = cfg["model"]

    device_str = cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(device_str)

    NUM_CLASSES   = mdl_cfg["num_classes"]
    INPUT_SIZE    = mdl_cfg["image_size"]
    MODEL_NAME    = out_cfg["model_name"]
    DATA_DIR      = dat_cfg["dir"]
    NUM_EPOCHS    = tr_cfg["epochs"]
    BATCH_SIZE    = tr_cfg["batch_size"]
    LR            = tr_cfg["learning_rate"]
    MIN_LR        = tr_cfg["min_lr"]
    WEIGHT_DECAY  = tr_cfg["weight_decay"]
    PATIENCE      = tr_cfg["patience"]
    WARMUP_EPOCHS = tr_cfg["warmup_epochs"]
    T_MAX         = tr_cfg["t_max"]
    LABEL_SMOOTH  = tr_cfg["label_smoothing"]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir  = os.path.join(out_cfg["dir"], f"{MODEL_NAME}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "run.log"),
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )
    logging.info(f"Config: {cfg}")

    print(f"Device: {DEVICE}  |  Variant: {mdl_cfg['variant']}  |  Input: {INPUT_SIZE}×{INPUT_SIZE}")

    # ---- Datasets ----
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    train_transform = build_train_transform(cfg)
    val_transform   = build_val_transform(cfg)

    full_train = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True,  download=True, transform=train_transform)
    test_ds    = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=val_transform)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train.targets)),
        test_size=dat_cfg["val_split"],
        random_state=42,
        stratify=full_train.targets,
    )
    train_subset = Subset(full_train, train_idx)
    val_subset   = Subset(full_train, val_idx)
    val_subset.dataset.transform = val_transform

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_ds)}")

    collate_fn = make_collate_fn(NUM_CLASSES, aug_cfg["mixup_alpha"], aug_cfg["cutmix_alpha"])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=dat_cfg["num_workers"], pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=dat_cfg["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,      batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=dat_cfg["num_workers"], pin_memory=True)

    class_names = full_train.classes
    print_classwise_counts(train_subset, "Training")
    print_classwise_counts(val_subset,   "Validation")

    # ---- Model ----
    model = UltraLightBlockNet_L1.from_variant(
        variant=mdl_cfg["variant"],
        num_classes=NUM_CLASSES,
        image_size=INPUT_SIZE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR)
    scaler    = GradScaler()

    # ---- Metrics storage ----
    records = []
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    best_embeddings = best_labels = None
    best_row = {}
    trajectory_params = []

    # ---- Training loop ----
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch + 1}: CPU {psutil.virtual_memory().percent}%")
            if torch.cuda.is_available():
                logging.info(f"Epoch {epoch + 1}: GPU {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        # Warmup
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {current_lr:.2e}")

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, tr_conf = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler, NUM_CLASSES
        )
        torch.cuda.empty_cache()

        va_loss, va_acc, va_prec, va_rec, va_f1, va_conf, va_emb, _, va_labels, va_inf = evaluate_model(
            model, val_loader, criterion, DEVICE, return_embeddings=True, num_classes=NUM_CLASSES
        )
        torch.cuda.empty_cache()

        trajectory_params.append({n: p.clone().detach().cpu() for n, p in model.named_parameters()})

        row = dict(
            epoch=epoch + 1,
            train_loss=tr_loss, train_acc=tr_acc, train_prec=tr_prec, train_rec=tr_rec, train_f1=tr_f1,
            val_loss=va_loss,   val_acc=va_acc,   val_prec=va_prec,   val_rec=va_rec,   val_f1=va_f1,
            inf_ms=va_inf,
        )
        records.append(row)

        print(
            f"Tr  loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"Val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
            f"inf={va_inf:.1f} ms"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            best_state = model.state_dict()
            best_row = row.copy()
            best_embeddings = va_emb
            best_labels = va_labels
            save_model(model, save_dir, f"best_{MODEL_NAME}")
            print(f"  ✓ New best val_acc = {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                save_model(model, save_dir, f"final_{MODEL_NAME}")
                break

    # ---- Restore best weights ----
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Test evaluation ----
    print("\n--- Test Set Evaluation ---")
    te_loss, te_acc, te_prec, te_rec, te_f1, te_conf, te_inf = evaluate_model(
        model, test_loader, criterion, DEVICE, return_embeddings=False, num_classes=NUM_CLASSES
    )
    print(f"Test loss={te_loss:.4f} acc={te_acc:.4f} prec={te_prec:.4f} rec={te_rec:.4f} f1={te_f1:.4f} inf={te_inf:.1f} ms")

    # ---- Save artefacts ----
    df = pd.DataFrame(records)
    df.rename(columns={"train_loss": "Train Loss", "val_loss": "Val Loss",
                        "train_acc": "Train Acc", "val_acc": "Val Acc"}, inplace=True)
    df.to_csv(os.path.join(save_dir, f"metrics_{MODEL_NAME}.csv"), index=False)

    pd.DataFrame([best_row]).to_csv(os.path.join(save_dir, f"best_metrics_{MODEL_NAME}.csv"), index=False)
    pd.DataFrame(te_conf).to_csv(os.path.join(save_dir, "test_confusion_matrix.csv"), index=False)

    plot_metrics(df, os.path.join(save_dir, f"curves_{MODEL_NAME}.png"))

    if best_embeddings is not None and len(best_embeddings) > 0:
        torch.save(best_embeddings, os.path.join(save_dir, "val_embeddings.pt"))
        plot_tsne(best_embeddings, best_labels, class_names=class_names,
                  save_path=os.path.join(save_dir, "tsne.png"))

    val_losses = [r["val_loss"] for r in records]
    plot_loss_landscape_with_library(
        model, criterion, val_loader, DEVICE,
        os.path.join(save_dir, "loss_trajectory.png"),
        trajectory_params=trajectory_params,
        val_losses=val_losses,
    )
    plot_loss_landscape_diagram(os.path.join(save_dir, "loss_landscape_3d.png"))

    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
