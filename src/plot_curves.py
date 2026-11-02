from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.visualization import save_curve_comparison, save_learning_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot learning curves from training_history.csv. "
        "One two-panel figure per run, plus optional overlays across runs."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run names under the checkpoints folder.")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels, same order as --runs.")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=[],
        help="Columns to overlay across runs, e.g. val_dice val_loss train_eval_dice",
    )
    parser.add_argument("--name", default=None, help="File prefix for overlay figures.")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--out-dir", default="reports/figures/curves")
    return parser.parse_args()


YLABELS = {
    "val_dice": "Validation Dice",
    "train_eval_dice": "Train Dice",
    "val_loss": "Validation loss",
    "train_eval_loss": "Train loss (eval mode)",
    "train_loss": "Train loss (augmented patches)",
    "lr": "Learning rate",
}


def main() -> None:
    args = parse_args()
    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must have one entry per run.")

    out_dir = Path(args.out_dir)
    histories = {}
    for run, label in zip(args.runs, labels):
        path = Path(args.checkpoints_dir) / run / "training_history.csv"
        if not path.exists():
            print(f"Skipping {run}: {path} not found")
            continue
        h = pd.read_csv(path)
        histories[label] = h
        out = out_dir / f"{run}_curves.png"
        save_learning_curves(h, out, title=run)
        print(f"Saved {out}")

    name = args.name or "_vs_".join(args.runs)[:80]
    for column in args.compare:
        out = out_dir / f"{name}_{column}.png"
        save_curve_comparison(histories, column, out, ylabel=YLABELS.get(column, column), title=name)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
