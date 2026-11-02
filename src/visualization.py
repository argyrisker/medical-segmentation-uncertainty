from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless servers (GPU nodes) have no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def class_names(num_classes: int) -> list[str]:
    """MSD Task04 label names (dataset.json)."""
    if num_classes == 3:
        return ["background", "anterior", "posterior"]
    return [f"class_{c}" for c in range(num_classes)]


def normalize_for_display(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def choose_slice(label: np.ndarray | None = None, score_map: np.ndarray | None = None) -> int:
    """
    Choose an axial slice. Prefer largest foreground label; otherwise highest score-map sum.
    """
    if label is not None and (label > 0).any():
        return int(np.argmax((label > 0).sum(axis=(0, 1))))
    if score_map is not None:
        return int(np.argmax(score_map.sum(axis=(0, 1))))
    return 0


def save_prediction_figure(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = choose_slice(label=label)
    img_slice = normalize_for_display(image[:, :, z])
    label_slice = label[:, :, z]
    pred_slice = pred[:, :, z]
    error_slice = (label_slice != pred_slice).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(np.rot90(img_slice), cmap="gray")
    axes[0].set_title("MRI")

    axes[1].imshow(np.rot90(img_slice), cmap="gray")
    axes[1].imshow(np.rot90(np.ma.masked_where(label_slice == 0, label_slice)), alpha=0.45)
    axes[1].set_title("Ground truth")

    axes[2].imshow(np.rot90(img_slice), cmap="gray")
    axes[2].imshow(np.rot90(np.ma.masked_where(pred_slice == 0, pred_slice)), alpha=0.45)
    axes[2].set_title("Prediction")

    axes[3].imshow(np.rot90(img_slice), cmap="gray")
    axes[3].imshow(np.rot90(np.ma.masked_where(error_slice == 0, error_slice)), alpha=0.55)
    axes[3].set_title("Error map")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_uncertainty_figure(
    image: np.ndarray,
    label: np.ndarray | None,
    pred: np.ndarray,
    uncertainty: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = choose_slice(label=label, score_map=uncertainty)
    img_slice = normalize_for_display(image[:, :, z])
    pred_slice = pred[:, :, z]
    unc_slice = uncertainty[:, :, z]

    n_cols = 4 if label is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(np.rot90(img_slice), cmap="gray")
    axes[0].set_title("MRI")

    if label is not None:
        label_slice = label[:, :, z]
        axes[1].imshow(np.rot90(img_slice), cmap="gray")
        axes[1].imshow(np.rot90(np.ma.masked_where(label_slice == 0, label_slice)), alpha=0.45)
        axes[1].set_title("Ground truth")
        pred_ax = axes[2]
        unc_ax = axes[3]
    else:
        pred_ax = axes[1]
        unc_ax = axes[2]

    pred_ax.imshow(np.rot90(img_slice), cmap="gray")
    pred_ax.imshow(np.rot90(np.ma.masked_where(pred_slice == 0, pred_slice)), alpha=0.45)
    pred_ax.set_title("Prediction")

    unc_ax.imshow(np.rot90(img_slice), cmap="gray")
    im = unc_ax.imshow(np.rot90(unc_slice), alpha=0.55)
    unc_ax.set_title("Uncertainty")
    fig.colorbar(im, ax=unc_ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_reliability_diagram(
    rows: list[dict],
    ece: float,
    out_path: str | Path,
    extra: list[tuple[list[dict], float, str]] | None = None,
    label: str = "model",
    title: str = "Reliability diagram",
) -> None:
    """
    Points below the diagonal mean over-confidence (confidence > accuracy).
    `extra` adds more curves, e.g. after temperature scaling: [(rows, ece, label)].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for r, e, name in [(rows, ece, label)] + list(extra or []):
        pts = [(x["confidence"], x["accuracy"]) for x in r if not np.isnan(x["confidence"])]
        xs, ys = zip(*pts) if pts else ([], [])
        ax.plot(xs, ys, marker="o", label=f"{name} (ECE={e:.4f})")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, names: list[str], out_path: str | Path, title: str = "") -> None:
    """Row-normalised colours (recall per true class), raw voxel counts as text."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=float)
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if row_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{row_norm[i, j]:.3f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8, color=colour)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_learning_curves(history: pd.DataFrame, out_path: str | Path, title: str = "") -> None:
    """
    Left: losses. Right: Dice. The best validation epoch is marked.
    train_loss is measured on augmented patches with dropout on; train_eval_*
    and val_* use the same whole-volume, eval-mode procedure and are the
    pair to compare for over-fitting.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = history
    v = h.dropna(subset=["val_dice"]) if "val_dice" in h else h.iloc[0:0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], color="grey", alpha=0.5, label="train loss (augmented patches)")
    if "train_eval_loss" in v and v["train_eval_loss"].notna().any():
        ax.plot(v["epoch"], v["train_eval_loss"], color="C0", label="train loss (whole volumes, eval mode)")
    if "val_loss" in v and v["val_loss"].notna().any():
        ax.plot(v["epoch"], v["val_loss"], color="C1", label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    ax = axes[1]
    if "train_eval_dice" in v and v["train_eval_dice"].notna().any():
        ax.plot(v["epoch"], v["train_eval_dice"], color="C0", label="train Dice")
    if len(v):
        ax.plot(v["epoch"], v["val_dice"], color="C1", label="validation Dice")
        best = v.loc[v["val_dice"].idxmax()]
        ax.axvline(best["epoch"], color="grey", linestyle=":", label=f"best val epoch {int(best['epoch'])}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean foreground Dice")
    ax.set_title("Dice")
    ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_curve_comparison(
    histories: dict[str, pd.DataFrame],
    column: str,
    out_path: str | Path,
    ylabel: str | None = None,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in histories.items():
        if column not in h:
            continue
        d = h.dropna(subset=[column])
        ax.plot(d["epoch"], d[column], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or column)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], out_path: str | Path, title: str = "") -> None:
    """curves: {label: (fpr, tpr, auc)}"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    ax.set_xlabel("False positive rate (correct voxels flagged)")
    ax.set_ylabel("True positive rate (error voxels flagged)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
