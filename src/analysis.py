from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_utils import load_config
from src.visualization import save_scatter_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def corr_safe(x, y) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    metrics_dir = Path(cfg["paths"]["metrics_dir"]) / args.run_name
    figures_dir = Path(cfg["paths"]["figures_dir"]) / args.run_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    uncertainty_path = metrics_dir / f"{args.split}_uncertainty.csv"
    if not uncertainty_path.exists():
        raise FileNotFoundError(f"Missing uncertainty file: {uncertainty_path}")

    df = pd.read_csv(uncertainty_path)

    pairs = [
        ("entropy_foreground_mean", "dice_mean", "Foreground entropy", "Mean Dice"),
        ("entropy_error_mean", "dice_mean", "Error-region entropy", "Mean Dice"),
        ("variance_foreground_mean", "dice_mean", "Foreground variance", "Mean Dice"),
        ("entropy_foreground_mean", "hd95_mean", "Foreground entropy", "Mean HD95"),
    ]

    rows = []

    for x_col, y_col, x_label, y_label in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        r = corr_safe(x, y)

        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "pearson_r": r,
                "n_cases": int((np.isfinite(x) & np.isfinite(y)).sum()),
            }
        )

        save_scatter_plot(
            x=x,
            y=y,
            xlabel=x_label,
            ylabel=y_label,
            title=f"{y_label} vs {x_label} | r={r:.3f}",
            out_path=figures_dir / f"{args.split}_{y_col}_vs_{x_col}.png",
        )

    out_csv = metrics_dir / f"{args.split}_dice_uncertainty_correlations.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved analysis correlations: {out_csv}")
    print(f"Saved plots: {figures_dir}")


if __name__ == "__main__":
    main()
