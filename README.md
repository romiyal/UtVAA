# UtVAA : Ultra-tiny Vision Transformer with Affix Attention 

**Ultra-tiny Vision Transformer with Affix Attention for Mobile Image Classification**

[![PyPI](https://img.shields.io/pypi/v/utvaa.svg)](https://pypi.org/project/utvaa/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

UtVAA is an ultra-lightweight neural network that fuses **depthwise dilated convolutions** with a novel **AffixAttentionBlock** — a hybrid module that combines linear-complexity self-attention (inspired by MobileViTv2) with **Coordinate Attention** for spatially-aware channel recalibration. The architecture achieves competitive accuracy on CIFAR-100 under strict parameter and FLOPs budgets, making it suitable for resource-constrained inference.

### Key Properties

| Property             | Detail                                   |
| -------------------- | ---------------------------------------- |
| Input resolution     | 256 × 256                                |
| Dataset              | CIFAR-100 (100 classes)                  |
| Attention complexity | Linear O(N)                              |
| Augmentation         | CutMix + MixUp + RandAugment             |
| Training strategy    | AdamW + Linear Warmup + Cosine Annealing |

---

## Architecture

![UtVAA Overall Architecture](assets/architecture.png)

```
Input (3 × 256 × 256)
      │
   Stem ──► ConvLayer [stride 2] ──► 128 × 128
      │
Stage 1 ──► DilatedBottleneck × 3 ──► 32 × 32
      │
Stage 2 ──► DilatedBottleneck (dilation=2)
         ──► AffixAttentionBlock (depth=2) ──► 16 × 16
      │
Stage 3 ──► DilatedBottleneck × 2 (dilation=4)
         ──► AffixAttentionBlock (depth=3) ──► 16 × 16
      │
   Head ──► ConvLayer + GlobalAvgPool ──► 1 × 1
      │
Classifier ──► Dropout + Linear ──► num_classes
```

**AffixAttentionBlock** — the core hybrid building block:

1. **LocalRepresentationBlock** — depthwise + pointwise convolutions project features to Transformer dimension.
2. Features are unfolded into non-overlapping 2 × 2 patches.
3. **Linear Self-Attention** — O(N) patch-wise attention via context vectors (MobileViTv2-style).
4. Patches are folded back; a 1 × 1 conv projects to the original channel dimension.
5. **Coordinate Attention** — applied independently to the skip-connection branch to inject spatial priors.
6. Both branches are concatenated and fused via a 1 × 1 conv, followed by a residual connection.

---

## Model Variants

| Variant    | Dims       | Channels              | Params (approx.) |
| ---------- | ---------- | --------------------- | ---------------- |
| **Tiny**   | [48, 64]   | [8, 16, 32, 48, 288]  | ~0.6 M           |
| **Medium** | [96, 112]  | [8, 32, 64, 96, 352]  | ~1.8 M           |
| **Large**  | [112, 132] | [8, 32, 72, 128, 416] | ~3.1 M           |
| **XLarge** | [112, 132] | [8, 32, 72, 198, 616] | ~5.4 M           |

---

## Repository Structure

```
UtVAA/
├── configs/
│   ├── cifar100.yaml          # CIFAR-100 hyperparameters
│   └── custom_dataset.yaml    # Custom pre-split dataset hyperparameters
├── utvaa/
│   ├── models/
│   │   └── ultralight_net.py  # UltraLightBlockNet_L1 and all sub-modules
│   ├── data/
│   │   └── datasets.py        # DataLoader utilities + TransformDataset
│   ├── engine/
│   │   └── trainer.py         # train_one_epoch / evaluate_model
│   └── utils/
│       └── helpers.py         # Checkpointing, plotting, t-SNE
├── assets/
│   └── architecture.png       # Overall architecture diagram
├── train.py                   # CIFAR-100 training entry point
├── train_custom.py            # Custom pre-split dataset training entry point
├── evaluate.py                # Checkpoint evaluation entry point
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Installation

### From PyPI (recommended)

```bash
pip install utvaa
```

### From source

```bash
# 1. Clone
git clone https://github.com/nishan-charlie/Utvaa.git
cd Utvaa

# 2. Create environment
conda create -n utvaa python=3.10 -y
conda activate utvaa

# 3. Install in editable mode (includes all dependencies)
pip install -e ".[dev]"
```

---

## Dataset Preparation

CIFAR-100 is downloaded automatically on first run via `torchvision.datasets.CIFAR100`.
Set `data.dir` in `configs/cifar100.yaml` to the directory where it should be stored,
or pass `--data-dir /path/to/cifar100` on the command line.

```
/path/to/cifar100/
└── cifar-100-python/     ← created automatically by torchvision
```

---

## Training

### CIFAR-100

```bash
# Train with default Large variant
python train.py --config configs/cifar100.yaml --data-dir /path/to/cifar100

# Train a specific variant
python train.py --config configs/cifar100.yaml \
                --data-dir /path/to/cifar100 \
                --variant medium \
                --epochs 600 \
                --output-dir ./outputs

# Specify GPU
python train.py --config configs/cifar100.yaml \
                --data-dir /path/to/cifar100 \
                --device cuda:1
```

### Custom pre-split dataset

For datasets pre-saved as `train_dataset.pt` / `val_dataset.pt` / `test_dataset.pt`
(PyTorch `Subset` objects via `torch.save`).  Early stopping tracks **validation loss**.

```bash
python train_custom.py --config configs/custom_dataset.yaml \
                       --data-dir /path/to/custom/data

python train_custom.py --config configs/custom_dataset.yaml \
                       --data-dir /path/to/custom/data \
                       --variant large --device cuda:0
```

Checkpoints, metrics CSV, t-SNE plots, and loss-curve figures are saved under `outputs/<model_name>_<timestamp>/`.

### Key Hyperparameters

#### `configs/cifar100.yaml`

| Parameter                   | Default | Description                                        |
| --------------------------- | ------- | -------------------------------------------------- |
| `model.variant`             | `large` | Architecture size (tiny / medium / large / xlarge) |
| `training.epochs`           | 1000    | Maximum epochs (early stopping applies)            |
| `training.batch_size`       | 96      | Mini-batch size                                    |
| `training.learning_rate`    | 1e-4    | Peak learning rate                                 |
| `training.patience`         | 200     | Early-stopping patience (val accuracy)             |
| `training.warmup_epochs`    | 10      | Linear warm-up epochs                              |
| `training.t_max`            | 200     | Cosine annealing period                            |
| `augmentation.mixup_alpha`  | 2.5     | MixUp α                                            |
| `augmentation.cutmix_alpha` | 3.5     | CutMix α                                           |

#### `configs/custom_dataset.yaml`

| Parameter                      | Default | Description                            |
| ------------------------------ | ------- | -------------------------------------- |
| `model.variant`                | `tiny`  | Architecture size                      |
| `training.epochs`              | 400     | Maximum epochs                         |
| `training.batch_size`          | 32      | Mini-batch size                        |
| `training.learning_rate`       | 1e-3    | Peak learning rate                     |
| `training.patience`            | 30      | Early-stopping patience (val loss)     |
| `training.val_batch_multiplier`| 3       | val/test batch = batch × this          |

---

## Evaluation

```bash
python evaluate.py \
    --config configs/cifar100.yaml \
    --checkpoint outputs/<run>/best_UltraLightBlockNet_L1.pth \
    --data-dir /path/to/cifar100 \
    --split test
```

---

## Training Output

Each training run produces the following artefacts:

```
outputs/<model>_<timestamp>/
├── best_<model>.pth           # Best checkpoint (by val accuracy)
├── final_<model>.pth          # Checkpoint at early-stop (if triggered)
├── metrics_<model>.csv        # Per-epoch train/val metrics
├── best_metrics_<model>.csv   # Metrics at best epoch
├── test_confusion_matrix.csv  # 100 × 100 confusion matrix
├── curves_<model>.png         # Loss + accuracy curves
├── tsne.png                   # t-SNE of validation embeddings
├── val_embeddings.pt          # Saved embedding tensor
├── loss_trajectory.png        # Validation loss over epochs
├── loss_landscape_3d.png      # 3-D loss surface (conceptual)
└── run.log                    # Memory and runtime log
```

---

## Using the Model Programmatically

```python
import torch
from utvaa import UltraLightBlockNet_L1

# Instantiate a named variant
model = UltraLightBlockNet_L1.from_variant("large", num_classes=100, image_size=256)
model.eval()

dummy = torch.randn(1, 3, 256, 256)
logits = model(dummy)          # (1, 100)
print(logits.shape)

# Or specify dims/channels manually
model = UltraLightBlockNet_L1(
    num_classes=100,
    image_size=256,
    dims=[112, 132],
    channels=[8, 32, 72, 128, 416],
)
```

---

## Hardware

Experiments were conducted on an NVIDIA GPU cluster. Training the Large variant on CIFAR-100 for 1000 epochs (with early stopping) takes approximately 18–24 hours on a single A100 GPU.

---

## Publishing to PyPI

```bash
# 1. Install build tools
pip install build twine

# 2. Build source distribution and wheel
python -m build

# 3. Upload to TestPyPI first (recommended sanity check)
twine upload --repository testpypi dist/*

# 4. Verify the install
pip install --index-url https://test.pypi.org/simple/ utvaa

# 5. Publish to PyPI
twine upload dist/*
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

The linear self-attention mechanism is adapted from [MobileViTv2](https://arxiv.org/abs/2206.02680).
Coordinate Attention is adapted from [Hou et al., 2021](https://arxiv.org/abs/2103.02907).
