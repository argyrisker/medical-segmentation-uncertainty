from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader

from src.config_utils import get_device, load_config, set_global_seed
from src.datasets import MSDConfig, ensure_msd_downloaded, get_labelled_cases, make_monai_dataset, split_cases
from src.inference_utils import predict_probabilities
from src.metrics import ece_score
from src.models import build_model
from src.transforms import get_val_transforms
from src.visualization import save_reliability_diagram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-voxels-per-case", type=int, default=200000)
    return parser.parse_args()


def sample_voxels(conf: np.ndarray, correct: np.ndarray, max_voxels: int, seed: int):
    conf = conf.reshape(-1)
    correct = correct.reshape(-1)
    if len(conf) <= max_voxels:
        return conf, correct
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(conf), size=max_voxels, replace=False)
    return conf[idx], correct[idx]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    run_name = args.run_name or checkpoint.get("run_name") or Path(args.checkpoint).parent.name

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    msd_cfg = MSDConfig(
        root_dir=cfg["dataset"]["root_dir"],
        task=cfg["dataset"]["task"],
        seed=seed,
    )
    ensure_msd_downloaded(msd_cfg, download=args.download)
    cases = get_labelled_cases(msd_cfg)
    _, val_cases, test_cases = split_cases(
        cases=cases,
        seed=seed,
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
    )
    eval_cases = val_cases if args.split == "val" else test_cases

    tf = get_val_transforms(
        pixdim=cfg["preprocessing"]["pixdim"],
        crop_margin=cfg["preprocessing"]["crop_margin"],
    )
    ds = make_monai_dataset(eval_cases, tf, cache_rate=0.0, num_workers=cfg["training"]["num_workers"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    all_conf = []
    all_correct = []
    all_conf_fg = []
    all_correct_fg = []

    for case_idx, batch in enumerate(loader):
        image = batch["image"]
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        probs = predict_probabilities(
            model=model,
            image=image,
            roi_size=tuple(cfg["inference"]["roi_size"]),
            sw_batch_size=cfg["inference"]["sw_batch_size"],
            num_classes=cfg["model"]["num_classes"],
            device=device,
            overlap=cfg["inference"]["overlap"],
        ).detach().cpu().numpy()[0]

        pred = probs.argmax(axis=0)
        conf = probs.max(axis=0)
        correct = pred == label

        conf_s, correct_s = sample_voxels(conf, correct, args.max_voxels_per_case, seed + case_idx)
        all_conf.append(conf_s)
        all_correct.append(correct_s)

        fg = np.logical_or(label > 0, pred > 0)
        if fg.any():
            conf_fg_s, correct_fg_s = sample_voxels(conf[fg], correct[fg], args.max_voxels_per_case, seed + 1000 + case_idx)
            all_conf_fg.append(conf_fg_s)
            all_correct_fg.append(correct_fg_s)

    conf_all = np.concatenate(all_conf)
    correct_all = np.concatenate(all_correct)

    ece_all, rows_all = ece_score(conf_all, correct_all, n_bins=cfg["calibration"]["n_bins"])

    out_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    fig_dir = Path(cfg["paths"]["figures_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows_all).to_csv(out_dir / f"{args.split}_calibration_all_voxels.csv", index=False)
    save_reliability_diagram(rows_all, ece_all, fig_dir / f"{args.split}_reliability_all_voxels.png")

    summary = {"split": args.split, "ece_all_voxels": ece_all}

    if all_conf_fg:
        conf_fg = np.concatenate(all_conf_fg)
        correct_fg = np.concatenate(all_correct_fg)
        ece_fg, rows_fg = ece_score(conf_fg, correct_fg, n_bins=cfg["calibration"]["n_bins"])
        pd.DataFrame(rows_fg).to_csv(out_dir / f"{args.split}_calibration_foreground.csv", index=False)
        save_reliability_diagram(rows_fg, ece_fg, fig_dir / f"{args.split}_reliability_foreground.png")
        summary["ece_foreground"] = ece_fg

    pd.DataFrame([summary]).to_csv(out_dir / f"{args.split}_calibration_summary.csv", index=False)

    print("Calibration summary:")
    print(summary)


if __name__ == "__main__":
    main()
