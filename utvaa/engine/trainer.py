"""
Training and evaluation loops for UltraLightBlockNet.

Supports CutMix/MixUp mixed labels, optional mixed-precision via GradScaler,
and gradient accumulation.
"""

import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scaler=None, num_classes: int = 100):
    """
    Run one training epoch.

    Args:
        model:        Neural network.
        train_loader: DataLoader whose collate_fn may return CutMix/MixUp tuple labels.
        criterion:    Loss function (e.g. CrossEntropyLoss with label_smoothing).
        optimizer:    Optimiser instance.
        device:       Torch device.
        scaler:       GradScaler for AMP.  Pass ``None`` to disable AMP.
        num_classes:  Number of output classes.

    Returns:
        ``(avg_loss, accuracy, macro_precision, macro_recall, macro_f1, confusion_matrix)``
    """
    model.to(device).train()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    accum_steps = 4
    optimizer.zero_grad()

    use_amp = scaler is not None and device.type == "cuda"

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        if isinstance(labels, tuple):
            target_a, target_b, lam = labels
            target_a = _to_hard_labels(target_a, device)
            target_b = _to_hard_labels(target_b, device)
        else:
            target_a = _to_hard_labels(labels, device)
            target_b, lam = None, None

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = _compute_loss(criterion, outputs, target_a, target_b, lam)
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = _compute_loss(criterion, outputs, target_a, target_b, lam)
            (loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        if target_b is not None:
            correct += (lam * (preds == target_a).float() + (1 - lam) * (preds == target_b).float()).sum().item()
            total += target_a.size(0)
            confusion += confusion_matrix(target_a.cpu().numpy(), preds.cpu().numpy(), labels=range(num_classes))
        else:
            correct += (preds == target_a).sum().item()
            total += target_a.size(0)
            confusion += confusion_matrix(target_a.cpu().numpy(), preds.cpu().numpy(), labels=range(num_classes))

    return _aggregate(total_loss, correct, total, confusion, len(train_loader))


def evaluate_model(model, val_loader, criterion, device,
                   return_embeddings: bool = False, num_classes: int = 100):
    """
    Evaluate the model on a validation or test split.

    Args:
        model:             Neural network.
        val_loader:        Evaluation DataLoader.
        criterion:         Loss function.
        device:            Torch device.
        return_embeddings: If ``True``, also return pooled feature vectors.
        num_classes:       Number of output classes.

    Returns:
        Without embeddings:
            ``(avg_loss, accuracy, precision, recall, f1, confusion, avg_inference_ms)``
        With embeddings:
            ``(avg_loss, accuracy, precision, recall, f1, confusion,
               embeddings_tensor, img_paths, labels_list, avg_inference_ms)``
    """
    model.to(device).eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    embeddings, img_paths, labels_list = [], [], []
    total_inference_time = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, batch_labels in val_loader:
            images = images.to(device)
            batch_labels = _to_hard_labels(batch_labels, device)

            t0 = time.time()
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            total_inference_time += time.time() - t0

            total_loss += criterion(outputs, batch_labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            confusion += confusion_matrix(batch_labels.cpu().numpy(), preds.cpu().numpy(), labels=range(num_classes))

            if return_embeddings:
                if use_amp:
                    with autocast():
                        feats = _extract_features(model, images)
                else:
                    feats = _extract_features(model, images)
                embeddings.append(feats.cpu())
                labels_list.extend(batch_labels.cpu().tolist())
                if hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'samples'):
                    indices = val_loader.dataset.indices
                    img_paths.extend(val_loader.dataset.dataset.samples[idx][0] for idx in indices)

    avg_loss, accuracy, precision, recall, f1, _ = _aggregate(
        total_loss, correct, total, confusion, len(val_loader)
    )
    avg_inf_ms = (total_inference_time / total) * 1000 if total > 0 else 0.0

    if return_embeddings:
        emb_tensor = torch.cat(embeddings, dim=0) if embeddings else torch.empty((0,))
        return avg_loss, accuracy, precision, recall, f1, confusion, emb_tensor, img_paths, labels_list, avg_inf_ms

    return avg_loss, accuracy, precision, recall, f1, confusion, avg_inf_ms


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_loss(criterion, outputs, target_a, target_b, lam):
    if target_b is not None:
        return lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
    return criterion(outputs, target_a)


def _to_hard_labels(labels, device):
    labels = labels.to(device)
    if labels.ndim == 2:
        return labels.argmax(dim=1)
    return labels


def _extract_features(model, images):
    x = model.stem(images)
    x = model.stage1(x)
    x = model.stage2(x)
    x = model.stage3(x)
    x = model.head(x)
    return x.view(x.size(0), -1)


def _aggregate(total_loss, correct, total, confusion, n_batches):
    accuracy = correct / total if total > 0 else 0.0
    tp = np.diag(confusion)
    precision = (tp / np.maximum(confusion.sum(axis=0), 1)).mean()
    recall    = (tp / np.maximum(confusion.sum(axis=1), 1)).mean()
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return total_loss / n_batches, accuracy, precision, recall, f1, confusion
