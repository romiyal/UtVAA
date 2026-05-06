"""
Standalone evaluation script for UltraLightBlockNet.

Usage:
    python evaluate.py --config configs/cifar100.yaml \
                       --checkpoint outputs/.../best_UltraLightBlockNet_L1.pth \
                       --data-dir /path/to/cifar100 \
                       --split test
"""

import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from utvaa.engine import evaluate_model
from utvaa.models import UltraLightBlockNet_L1


CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD  = [0.2673, 0.2564, 0.2762]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained UltraLightBlockNet checkpoint")
    p.add_argument("--config",     required=True,  help="YAML config path")
    p.add_argument("--checkpoint", required=True,  help="Path to .pth model checkpoint")
    p.add_argument("--data-dir",   default=None,   help="Override data.dir in config")
    p.add_argument("--split",      default="test", choices=["val", "test"],
                   help="Dataset split to evaluate on")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mdl_cfg = cfg["model"]
    dat_cfg = cfg["data"]

    data_dir   = args.data_dir or dat_cfg["dir"]
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE     = torch.device(device_str)
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    INPUT_SIZE = mdl_cfg["image_size"]
    NUM_CLASSES = mdl_cfg["num_classes"]

    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS, antialias=True),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    is_train_split = args.split == "val"
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=is_train_split, download=True, transform=val_transform
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=dat_cfg["num_workers"], pin_memory=True,
    )

    model = UltraLightBlockNet_L1.from_variant(
        variant=mdl_cfg["variant"],
        num_classes=NUM_CLASSES,
        image_size=INPUT_SIZE,
    ).to(DEVICE)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    print(f"Loaded checkpoint: {args.checkpoint}")

    criterion = nn.CrossEntropyLoss()
    loss, acc, prec, rec, f1, conf, inf_ms = evaluate_model(
        model, loader, criterion, DEVICE, return_embeddings=False, num_classes=NUM_CLASSES
    )

    print(f"\n{args.split.capitalize()} Results")
    print(f"  Loss      : {loss:.4f}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Inf. time : {inf_ms:.2f} ms/image")


if __name__ == "__main__":
    main()
