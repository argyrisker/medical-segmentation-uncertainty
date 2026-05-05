from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def save_reliability_diagram(rows: list[dict], ece: float, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [r["confidence"] for r in rows if not np.isnan(r["confidence"])]
    ys = [r["accuracy"] for r in rows if not np.isnan(r["accuracy"])]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.scatter(xs, ys)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram | ECE={ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
