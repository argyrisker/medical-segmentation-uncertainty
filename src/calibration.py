from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from torch.nn import functional as F

from src.config_utils import config_from_checkpoint, get_device, set_global_seed
from src.datasets import get_splits, make_monai_dataset
from src.inference_utils import inference_kwargs, predict_logits
from src.metrics import ece_score
from src.models import build_model
from src.transforms import build_eval_transforms
from src.visualization import save_reliability_diagram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration analysis, optionally with temperature scaling.")
    parser.add_argument("--config", default=None, help="Only used if the checkpoint has no stored config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-voxels-per-case", type=int, default=200000)
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Fit one temperature on validation foreground-region voxels, then report the split before/after.",
    )
    return parser.parse_args()


def _sample_idx(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


@torch.no_grad()
def collect_logits(model, cases, cfg, device, max_voxels: int, seed: int) -> dict[str, np.ndarray]:
    """
    Sampled voxel logits and labels, over all voxels and over the foreground
    region (truth or prediction is foreground, where all errors live).
    """
    ds = make_monai_dataset(cases, build_eval_transforms(cfg), cache_rate=0.0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])
    rng = np.random.default_rng(seed)
    out = {"all_logits": [], "all_labels": [], "fg_logits": [], "fg_labels": []}

    model.eval()
    for batch in loader:
        logits = predict_logits(model, batch["image"], device=device, **inference_kwargs(cfg))
        logits = logits.detach().cpu().numpy()[0]  # C x H x W x D
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int64)
        flat_logits = logits.reshape(logits.shape[0], -1).T  # N x C
        flat_label = label.reshape(-1)
        pred = flat_logits.argmax(axis=1)

        idx = _sample_idx(len(flat_label), max_voxels, rng)
        out["all_logits"].append(flat_logits[idx])
        out["all_labels"].append(flat_label[idx])

        fg = np.flatnonzero((flat_label > 0) | (pred > 0))
        if len(fg):
            fg = fg[_sample_idx(len(fg), max_voxels, rng)]
            out["fg_logits"].append(flat_logits[fg])
            out["fg_labels"].append(flat_label[fg])

    return {k: np.concatenate(v) for k, v in out.items()}


def ece_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float, n_bins: int):
    probs = torch.softmax(torch.from_numpy(logits / temperature), dim=1).numpy()
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    return ece_score(conf, correct, n_bins=n_bins)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    return float(F.cross_entropy(torch.from_numpy(logits / temperature), torch.from_numpy(labels)).item())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Temperature scaling (Guo et al., 2017): divide logits by one scalar T
    chosen to minimise NLL on held-out data. T > 1 softens over-confident
    predictions. Argmax, and therefore Dice, is unchanged.
    """
    grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 300))
    losses = [nll(logits, labels, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(checkpoint, args.config)
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    run_name = args.run_name or checkpoint.get("run_name") or Path(args.checkpoint).parent.name
    n_bins = int(cfg["calibration"]["n_bins"])

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_cases, test_cases = get_splits(cfg, download=args.download)
    eval_cases = val_cases if args.split == "val" else test_cases

    out_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    fig_dir = Path(cfg["paths"]["figures_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = collect_logits(model, eval_cases, cfg, device, args.max_voxels_per_case, seed)

    ece_all, rows_all = ece_from_logits(data["all_logits"], data["all_labels"], 1.0, n_bins)
    ece_fg, rows_fg = ece_from_logits(data["fg_logits"], data["fg_labels"], 1.0, n_bins)
    summary = {
        "split": args.split,
        "ece_all_voxels": ece_all,
        "ece_foreground": ece_fg,
        "nll_foreground": nll(data["fg_logits"], data["fg_labels"], 1.0),
    }
    pd.DataFrame(rows_all).to_csv(out_dir / f"{args.split}_calibration_all_voxels.csv", index=False)
    pd.DataFrame(rows_fg).to_csv(out_dir / f"{args.split}_calibration_foreground.csv", index=False)

    extra_all, extra_fg = None, None
    if args.temperature_scaling:
        # Fit on validation only; never on the split being reported.
        val_data = data if args.split == "val" else collect_logits(
            model, val_cases, cfg, device, args.max_voxels_per_case, seed + 1
        )
        t = fit_temperature(val_data["fg_logits"], val_data["fg_labels"])
        ece_all_t, rows_all_t = ece_from_logits(data["all_logits"], data["all_labels"], t, n_bins)
        ece_fg_t, rows_fg_t = ece_from_logits(data["fg_logits"], data["fg_labels"], t, n_bins)
        summary.update(
            temperature=t,
            ece_all_voxels_scaled=ece_all_t,
            ece_foreground_scaled=ece_fg_t,
            nll_foreground_scaled=nll(data["fg_logits"], data["fg_labels"], t),
        )
        pd.DataFrame(rows_fg_t).to_csv(out_dir / f"{args.split}_calibration_foreground_scaled.csv", index=False)
        extra_all = [(rows_all_t, ece_all_t, f"temperature scaled (T={t:.2f})")]
        extra_fg = [(rows_fg_t, ece_fg_t, f"temperature scaled (T={t:.2f})")]

    save_reliability_diagram(
        rows_all, ece_all, fig_dir / f"{args.split}_reliability_all_voxels.png",
        extra=extra_all, label="softmax", title=f"{run_name} | {args.split} | all voxels",
    )
    save_reliability_diagram(
        rows_fg, ece_fg, fig_dir / f"{args.split}_reliability_foreground.png",
        extra=extra_fg, label="softmax", title=f"{run_name} | {args.split} | foreground region",
    )

    pd.DataFrame([summary]).to_csv(out_dir / f"{args.split}_calibration_summary.csv", index=False)
    print("Calibration summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
