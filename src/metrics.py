from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


EPS = 1e-8


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * np.logical_and(pred, target).sum() / (denom + EPS))


def sensitivity_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    if tp + fn == 0:
        return np.nan
    return float(tp / (tp + fn + EPS))


def specificity_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    if tn + fp == 0:
        return np.nan
    return float(tn / (tn + fp + EPS))


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError("Expected a 3D mask for Hausdorff distance.")
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    structure = generate_binary_structure(mask.ndim, 1)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def hd95(pred: np.ndarray, target: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """
    Symmetric 95th percentile Hausdorff distance.
    Returns NaN if either mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return np.nan

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if not pred_surface.any() or not target_surface.any():
        return np.nan

    dt_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    distances = np.concatenate(
        [
            dt_to_target[pred_surface],
            dt_to_pred[target_surface],
        ]
    )

    return float(np.percentile(distances, 95))


def segmentation_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    include_background: bool = False,
) -> dict[str, float]:
    """
    Compute class-wise and mean segmentation metrics.

    pred_labels and true_labels should be integer label maps with shape H x W x D.
    """
    rows: dict[str, float] = {}
    classes = range(num_classes) if include_background else range(1, num_classes)

    dice_values = []
    hd95_values = []
    sens_values = []
    spec_values = []

    for c in classes:
        pred_c = pred_labels == c
        true_c = true_labels == c

        d = dice_score(pred_c, true_c)
        h = hd95(pred_c, true_c, spacing=spacing)
        s = sensitivity_score(pred_c, true_c)
        sp = specificity_score(pred_c, true_c)

        rows[f"dice_class_{c}"] = d
        rows[f"hd95_class_{c}"] = h
        rows[f"sensitivity_class_{c}"] = s
        rows[f"specificity_class_{c}"] = sp

        dice_values.append(d)
        hd95_values.append(h)
        sens_values.append(s)
        spec_values.append(sp)

    rows["dice_mean"] = float(np.nanmean(dice_values))
    rows["hd95_mean"] = float(np.nanmean(hd95_values))
    rows["sensitivity_mean"] = float(np.nanmean(sens_values))
    rows["specificity_mean"] = float(np.nanmean(spec_values))

    return rows


def predictive_entropy(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    probabilities shape: C x H x W x D
    """
    return -np.sum(probabilities * np.log(probabilities + eps), axis=0)


def variation_ratio(probabilities_stack: np.ndarray) -> np.ndarray:
    """
    probabilities_stack shape: T x C x H x W x D
    """
    labels = probabilities_stack.argmax(axis=1)  # T x H x W x D
    out = np.zeros(labels.shape[1:], dtype=np.float32)
    for index in np.ndindex(out.shape):
        vals, counts = np.unique(labels[(slice(None),) + index], return_counts=True)
        out[index] = 1.0 - counts.max() / labels.shape[0]
    return out


def ece_score(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    confidence = confidence.reshape(-1)
    correct = correct.reshape(-1).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(confidence)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = int(mask.sum())
        if count == 0:
            acc = np.nan
            conf = np.nan
            gap = np.nan
            frac = 0.0
        else:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            gap = abs(acc - conf)
            frac = count / n
            ece += frac * gap

        rows.append(
            {
                "bin": i,
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "fraction": float(frac),
                "accuracy": acc,
                "confidence": conf,
                "gap": gap,
            }
        )

    return float(ece), rows
