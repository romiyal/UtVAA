"""
Training utility functions: model checkpointing, metric plots, t-SNE, loss landscape.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_model(model: torch.nn.Module, save_dir: str, filename: str):
    """Save model state dict to ``<save_dir>/<filename>.pth``."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{filename}.pth")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def save_embeddings(embeddings, save_path: str):
    """Save an embedding tensor to disk with ``torch.save``."""
    torch.save(embeddings, save_path)
    print(f"Embeddings saved to {save_path}")


# ---------------------------------------------------------------------------
# Training-curve plots
# ---------------------------------------------------------------------------

def plot_metrics(df, save_path: str):
    """
    Plot train/validation loss and accuracy from a metrics DataFrame.

    Expected columns: Train Loss, Val Loss, Train Acc, Val Acc.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, label, color in [("Train Loss", "Train", "steelblue"), ("Val Loss", "Val", "tomato")]:
        if col in df.columns:
            axes[0].plot(df[col], label=label, color=color, linewidth=1.5)
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for col, label, color in [("Train Acc", "Train", "steelblue"), ("Val Acc", "Val", "tomato")]:
        if col in df.columns:
            axes[1].plot(df[col], label=label, color=color, linewidth=1.5)
    axes[1].set_title("Accuracy", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot saved to {save_path}")


# ---------------------------------------------------------------------------
# t-SNE visualisation
# ---------------------------------------------------------------------------

def plot_tsne(embeddings, labels, class_names=None, save_path: str = "tsne.png"):
    """
    Reduce ``embeddings`` to 2-D with t-SNE and plot coloured by class.

    Args:
        embeddings:   Tensor or ndarray of shape (N, D).
        labels:       Integer class labels of length N.
        class_names:  Optional list of string class names.
        save_path:    Output PNG path.
    """
    from sklearn.manifold import TSNE

    if hasattr(embeddings, "numpy"):
        emb_np = embeddings.numpy()
    else:
        emb_np = np.array(embeddings)

    labels_np = np.array(labels)
    n_samples = min(5000, len(emb_np))
    if len(emb_np) > n_samples:
        idx = np.random.choice(len(emb_np), n_samples, replace=False)
        emb_np, labels_np = emb_np[idx], labels_np[idx]

    print("Running t-SNE …")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    proj = tsne.fit_transform(emb_np)

    n_classes = len(np.unique(labels_np))
    cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv")
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls_idx in np.unique(labels_np):
        mask = labels_np == cls_idx
        label_str = class_names[cls_idx] if class_names is not None else str(cls_idx)
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6,
                   color=colors[cls_idx], label=label_str)

    ax.set_title("t-SNE Feature Embedding", fontsize=14)
    ax.axis("off")
    if n_classes <= 20:
        ax.legend(fontsize=6, ncol=2, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Loss landscape
# ---------------------------------------------------------------------------

def plot_loss_landscape_with_library(
    _model,
    _criterion,
    _val_loader,
    _device,
    save_path: str,
    trajectory_params=None,
    val_losses=None,
):
    """
    Visualise the training loss trajectory.

    The first four arguments (model, criterion, val_loader, device) are reserved
    for a full 2-D filter-normalised landscape computation via the
    ``loss-landscapes`` library.  Currently, the function plots the per-epoch
    validation loss curve as a lightweight proxy.
    """
    del trajectory_params  # reserved for future filter-normalised landscape
    if val_losses is not None and len(val_losses) > 0:
        epochs = list(range(1, len(val_losses) + 1))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, val_losses, color="steelblue", linewidth=2)
        ax.fill_between(epochs, val_losses, alpha=0.15, color="steelblue")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title("Loss Trajectory", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss trajectory saved to {save_path}")
    else:
        print("No val_losses provided; skipping loss landscape plot.")


def plot_loss_landscape_diagram(save_path: str):
    """
    Generate a synthetic 3-D loss landscape surface for conceptual illustration.
    """
    import mpl_toolkits.mplot3d  # noqa: F401  — registers the "3d" projection

    x = np.linspace(-3, 3, 80)
    y = np.linspace(-3, 3, 80)
    X, Y = np.meshgrid(x, y)
    Z = (
        0.3 * (X ** 2 + Y ** 2)
        + 0.4 * np.sin(X * 1.5) * np.cos(Y * 1.5)
        + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.8)
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Direction 1", fontsize=11)
    ax.set_ylabel("Direction 2", fontsize=11)
    ax.set_zlabel("Loss", fontsize=11)
    ax.set_title("Loss Landscape (Conceptual)", fontsize=13)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss landscape diagram saved to {save_path}")
