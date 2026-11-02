from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.compare_runs import to_markdown
from src.visualization import save_roc_curves

LABELS = {"deterministic": "softmax (1 pass)", "mc_dropout": "MC dropout", "ensemble": "deep ensemble"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare uncertainty methods: table + overlaid error-detection ROC.")
    parser.add_argument(
        "--entries",
        nargs="+",
        required=True,
        help="run_name:mode pairs, e.g. baseline_seed42:deterministic baseline_ensemble:ensemble",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--name", required=True)
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs/tables")
    parser.add_argument("--fig-dir", default="reports/figures/uncertainty")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, curves = [], {}
    for entry in args.entries:
        run, mode = entry.split(":", 1)
        d = Path(args.metrics_dir) / run
        summary = d / f"{args.split}_uncertainty_{mode}_summary.csv"
        roc = d / f"{args.split}_roc_{mode}.csv"
        if not summary.exists():
            print(f"Skipping {entry}: {summary} not found")
            continue
        s = pd.read_csv(summary).iloc[0].to_dict()
        rows.append(s)
        if roc.exists():
            r = pd.read_csv(roc)
            curves[f"{LABELS.get(mode, mode)} [{run}]"] = (r["fpr"].values, r["tpr"].values, s["error_auroc_pooled"])

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.name}.csv", index=False)
    (out_dir / f"{args.name}.md").write_text(f"## {args.name}\n\n" + to_markdown(df))
    fig = Path(args.fig_dir) / f"{args.name}_roc.png"
    save_roc_curves(curves, fig, title=f"Detecting segmentation errors from uncertainty ({args.split})")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)
    print(f"\nSaved {out_dir / (args.name + '.csv')} and {fig}")


if __name__ == "__main__":
    main()
