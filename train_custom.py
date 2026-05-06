"""
Training entry point for UltraLightBlockNet on a custom pre-split dataset.

Expects the data directory to contain three PyTorch Subset files saved with
``torch.save``:  ``train_dataset.pt``, ``val_dataset.pt``, ``test_dataset.pt``.

Early stopping is based on **validation loss** (minimise), unlike the CIFAR-100
script which tracks validation accuracy.  No CutMix/MixUp or AMP are used.

Usage:
    python train_custom.py --config configs/custom_dataset.yaml \\
                           --data-dir /path/to/custom/data
    python train_custom.py --config configs/custom_dataset.yaml \\
                           --variant medium --device cuda:0
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utvaa.data import TransformDataset, print_classwise_counts
from utvaa.engine import evaluate_model, train_one_epoch
from utvaa.models import UltraLightBlockNet_L1
from utvaa.utils import (
    plot_metrics,
    plot_tsne,
    save_embeddings,
    save_model,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train UltraLightBlockNet on a custom pre-split dataset")
    p.add_argument("--config",     default="configs/custom_dataset.yaml")
    p.add_argument("--data-dir",   default=None, help="Override data.dir in config")
    p.add_argument("--output-dir", default=None, help="Override output.dir in config")
    p.add_argument("--variant",    default=None, choices=["tiny", "medium", "large", "xlarge"])
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--device",     default=None, help="e.g. cuda:0, cuda:1, cpu")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg, args):
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
# Transforms  (ImageNet statistics — no CIFAR-specific normalisation)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = merge_args(load_config(args.config), args)

    tr_cfg  = cfg["training"]
    dat_cfg = cfg["data"]
    out_cfg = cfg["output"]
    mdl_cfg = cfg["model"]

    device_str = cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(device_str)

    INPUT_SIZE   = mdl_cfg["image_size"]
    MODEL_NAME   = out_cfg["model_name"]
    DATA_DIR     = dat_cfg["dir"]
    NUM_EPOCHS   = tr_cfg["epochs"]
    BATCH_SIZE   = tr_cfg["batch_size"]
    VAL_BATCH    = BATCH_SIZE * tr_cfg.get("val_batch_multiplier", 3)
    LR           = tr_cfg["learning_rate"]
    MIN_LR       = tr_cfg["min_lr"]
    WEIGHT_DECAY = tr_cfg["weight_decay"]
    PATIENCE     = tr_cfg["patience"]
    T_MAX        = tr_cfg["t_max"]
    LABEL_SMOOTH = tr_cfg["label_smoothing"]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir  = os.path.join(out_cfg["dir"], f"{MODEL_NAME}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {DEVICE}  |  Variant: {mdl_cfg['variant']}  |  Input: {INPUT_SIZE}×{INPUT_SIZE}")

    # ---- Load pre-saved splits ----
    for fname in ("train_dataset.pt", "val_dataset.pt", "test_dataset.pt"):
        if not os.path.isfile(os.path.join(DATA_DIR, fname)):
            raise FileNotFoundError(f"Expected {os.path.join(DATA_DIR, fname)}")

    train_subset = torch.load(os.path.join(DATA_DIR, "train_dataset.pt"), weights_only=False)
    val_subset   = torch.load(os.path.join(DATA_DIR, "val_dataset.pt"),   weights_only=False)
    test_subset  = torch.load(os.path.join(DATA_DIR, "test_dataset.pt"),  weights_only=False)

    train_dataset = TransformDataset(train_subset, transform=build_train_transform(INPUT_SIZE))
    val_dataset   = TransformDataset(val_subset,   transform=build_val_transform(INPUT_SIZE))
    test_dataset  = TransformDataset(test_subset,  transform=build_val_transform(INPUT_SIZE))

    num_classes = len(train_subset.dataset.classes)
    class_names = train_subset.dataset.classes
    print(f"Classes: {num_classes}  |  {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=True,  num_workers=dat_cfg["num_workers"])
    val_loader   = DataLoader(val_dataset,   batch_size=VAL_BATCH,   shuffle=False, num_workers=dat_cfg["num_workers"])
    test_loader  = DataLoader(test_dataset,  batch_size=VAL_BATCH,   shuffle=False, num_workers=dat_cfg["num_workers"])

    print_classwise_counts(train_dataset, "Training")
    print_classwise_counts(val_dataset,   "Validation")
    print_classwise_counts(test_dataset,  "Test")

    # ---- Model ----
    model = UltraLightBlockNet_L1.from_variant(
        variant=mdl_cfg["variant"],
        num_classes=num_classes,
        image_size=INPUT_SIZE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine annealing steps every epoch, no warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR)

    # ---- Metrics storage ----
    records = []
    best_val_loss = float("inf")
    no_improve    = 0
    best_row      = {}
    best_embeddings = best_labels = None

    # ---- Training loop ----
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS}  LR: {optimizer.param_groups[0]['lr']:.2e} ---")

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scaler=None, num_classes=num_classes,
        )

        va_loss, va_acc, va_prec, va_rec, va_f1, va_conf, va_emb, _, va_labels, va_inf = evaluate_model(
            model, val_loader, criterion, DEVICE,
            return_embeddings=True, num_classes=num_classes,
        )

        scheduler.step()

        row = dict(
            epoch=epoch + 1,
            train_loss=tr_loss, train_acc=tr_acc, train_prec=tr_prec, train_rec=tr_rec, train_f1=tr_f1,
            val_loss=va_loss,   val_acc=va_acc,   val_prec=va_prec,   val_rec=va_rec,   val_f1=va_f1,
            inf_ms=va_inf,
        )
        records.append(row)

        improvement = best_val_loss - va_loss
        print(
            f"Tr  loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"Val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
            f"Δloss={improvement:+.4f} | inf={va_inf:.1f} ms"
        )

        if va_loss < best_val_loss:
            best_val_loss   = va_loss
            no_improve      = 0
            best_row        = row.copy()
            best_embeddings = va_emb
            best_labels     = va_labels
            save_model(model, save_dir, f"best_{MODEL_NAME}")
            print(f"  ✓ New best val_loss = {best_val_loss:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement: {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # ---- Test evaluation ----
    print("\n--- Test Set Evaluation ---")
    te_loss, te_acc, te_prec, te_rec, te_f1, te_conf, te_emb, _, te_labels, te_inf = evaluate_model(
        model, test_loader, criterion, DEVICE,
        return_embeddings=True, num_classes=num_classes,
    )
    print(
        f"Test  loss={te_loss:.4f} acc={te_acc:.4f} prec={te_prec:.4f} "
        f"rec={te_rec:.4f} f1={te_f1:.4f} inf={te_inf:.1f} ms"
    )

    # ---- Save artefacts ----
    df = pd.DataFrame(records)
    df.rename(columns={"train_loss": "Train Loss", "val_loss": "Val Loss",
                        "train_acc": "Train Acc", "val_acc": "Val Acc"}, inplace=True)

    plot_metrics(df, os.path.join(save_dir, f"curves_{MODEL_NAME}.png"))

    df_test = pd.DataFrame([{
        "epoch": "test", "Test Loss": te_loss, "Test Acc": te_acc,
        "Test Precision": te_prec, "Test Recall": te_rec, "Test F1": te_f1,
        "Inf ms": te_inf,
    }])
    pd.concat([df, pd.DataFrame([best_row]), df_test], ignore_index=True).to_csv(
        os.path.join(save_dir, f"metrics_{MODEL_NAME}.csv"), index=False
    )

    pd.DataFrame(te_conf).to_csv(os.path.join(save_dir, "test_confusion_matrix.csv"), index=False)

    for tag, emb, lbl in [("val", best_embeddings, best_labels), ("test", te_emb, te_labels)]:
        if emb is not None and len(emb) > 0:
            save_embeddings(emb, os.path.join(save_dir, f"{tag}_embeddings.pt"))
            plot_tsne(emb, lbl, class_names=class_names,
                      save_path=os.path.join(save_dir, f"tsne_{tag}.png"))

    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
