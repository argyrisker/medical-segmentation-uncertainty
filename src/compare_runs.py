from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise several training runs into one table (CSV + Markdown) for slides."
    )
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument(
        "--reference",
        default=None,
        help="Run to compare against: adds the Dice difference and a paired Wilcoxon p-value "
        "over validation cases (and test cases when test metrics exist).",
    )
    parser.add_argument("--name", required=True, help="Output file name, e.g. stage3_ablation")
    parser.add_argument("--group-seeds", action="store_true", help="Also average runs that differ only by _seedN.")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs/tables")
    return parser.parse_args()


def load_run(run: str, ckpt_dir: Path, metrics_dir: Path) -> dict:
    run_dir = ckpt_dir / run
    row: dict = {"run": run}
    summary_path = run_dir / "summary.json"
    hist_path = run_dir / "training_history.csv"

    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        row.update(
            n_train=s.get("n_train_cases"),
            params=s.get("trainable_parameters"),
            epochs=s.get("epochs"),
            best_epoch=s.get("best_epoch"),
            best_val_dice=s.get("best_val_dice"),
            elapsed_min=s.get("elapsed_min"),
        )
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        v = h.dropna(subset=["val_dice"])
        if len(v):
            best = v.loc[v["val_dice"].idxmax()]
            last = v.iloc[-1]
            row.setdefault("best_epoch", int(best["epoch"]))
            row.setdefault("best_val_dice", float(best["val_dice"]))
            if "train_eval_dice" in v:
                row["train_dice_at_best"] = float(best["train_eval_dice"])
                row["final_train_dice"] = float(last["train_eval_dice"])
                row["final_val_dice"] = float(last["val_dice"])
                # Generalisation gap at the end of training.
                row["final_gap"] = float(last["train_eval_dice"] - last["val_dice"])
            if "val_loss" in v and v["val_loss"].notna().any():
                row["min_val_loss_epoch"] = int(v.loc[v["val_loss"].idxmin(), "epoch"])
    test_path = metrics_dir / run / "test_metrics.csv"
    if test_path.exists():
        t = pd.read_csv(test_path)
        row["test_dice"] = float(t["dice_mean"].mean())
        row["test_dice_std"] = float(t["dice_mean"].std())
        row["test_hd95"] = float(t["hd95_mean"].mean())
    return row


def paired_test(a: pd.DataFrame, b: pd.DataFrame, column: str = "dice_mean") -> tuple[float, float]:
    m = a[["case_id", column]].merge(b[["case_id", column]], on="case_id", suffixes=("_a", "_b"))
    diff = m[f"{column}_a"] - m[f"{column}_b"]
    if len(m) < 5 or np.allclose(diff, 0):
        return float(diff.mean()) if len(m) else np.nan, np.nan
    return float(diff.mean()), float(wilcoxon(m[f"{column}_a"], m[f"{column}_b"]).pvalue)


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(x, column: str):
        if isinstance(x, float):
            if np.isnan(x):
                return ""
            if column.endswith("_p") and x < 1e-4:
                return "<0.0001"
            return f"{x:.4f}"
        return "" if x is None else str(x)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(x, c) for c, x in zip(df.columns, r.values)) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ckpt_dir, metrics_dir = Path(args.checkpoints_dir), Path(args.metrics_dir)
    rows = [load_run(r, ckpt_dir, metrics_dir) for r in args.runs]

    if args.reference:
        ref_val = ckpt_dir / args.reference / "val_per_case_best.csv"
        ref_test = metrics_dir / args.reference / "test_metrics.csv"
        for row in rows:
            if row["run"] == args.reference:
                continue
            run_val = ckpt_dir / row["run"] / "val_per_case_best.csv"
            if ref_val.exists() and run_val.exists():
                d, p = paired_test(pd.read_csv(run_val), pd.read_csv(ref_val))
                row["val_dice_diff_vs_ref"], row["val_wilcoxon_p"] = d, p
            run_test = metrics_dir / row["run"] / "test_metrics.csv"
            if ref_test.exists() and run_test.exists():
                d, p = paired_test(pd.read_csv(run_test), pd.read_csv(ref_test))
                row["test_dice_diff_vs_ref"], row["test_wilcoxon_p"] = d, p

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.name}.csv", index=False)
    md = f"## {args.name}\n\n" + to_markdown(df)

    if args.group_seeds:
        df["experiment"] = df["run"].str.replace(r"_seed\d+$", "", regex=True)
        numeric = [c for c in ["best_val_dice", "final_gap", "test_dice", "test_hd95"] if c in df]
        g = df.groupby("experiment")[numeric].agg(["mean", "std", "count"])
        g.columns = ["_".join(c) for c in g.columns]
        g = g.reset_index()
        g.to_csv(out_dir / f"{args.name}_by_experiment.csv", index=False)
        md += "\n### Averaged over seeds\n\n" + to_markdown(g)

    (out_dir / f"{args.name}.md").write_text(md)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.drop(columns=[c for c in ["experiment"] if c in df]))
    print(f"\nSaved {out_dir / (args.name + '.csv')} and .md")


if __name__ == "__main__":
    main()
