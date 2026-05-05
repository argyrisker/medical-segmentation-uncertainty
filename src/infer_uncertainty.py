from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference

from src.config_utils import get_device, load_config, set_global_seed
from src.datasets import MSDConfig, ensure_msd_downloaded, get_labelled_cases, make_monai_dataset, split_cases
from src.metrics import predictive_entropy, segmentation_metrics
from src.models import build_model, enable_mc_dropout
from src.transforms import get_val_transforms
from src.visualization import save_uncertainty_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-vis", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def mc_dropout_probs(model, image, cfg, device, num_samples: int) -> np.ndarray:
    model.eval()
    enable_mc_dropout(model)

    image = image.to(device)
    probs_list = []

    for _ in range(num_samples):
        logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(cfg["inference"]["roi_size"]),
            sw_batch_size=cfg["inference"]["sw_batch_size"],
            predictor=model,
            overlap=cfg["inference"]["overlap"],
        )
        probs = torch.softmax(logits, dim=1)
        probs_list.append(probs.detach().cpu().numpy()[0])

    return np.stack(probs_list, axis=0)  # T x C x H x W x D


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg["project"]["seed"]))
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    run_name = args.run_name or checkpoint.get("run_name") or Path(args.checkpoint).parent.name

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_samples = args.num_samples or cfg["uncertainty"]["mc_samples"]

    msd_cfg = MSDConfig(
        root_dir=cfg["dataset"]["root_dir"],
        task=cfg["dataset"]["task"],
        seed=cfg["project"]["seed"],
    )
    ensure_msd_downloaded(msd_cfg, download=args.download)
    cases = get_labelled_cases(msd_cfg)
    train_cases, val_cases, test_cases = split_cases(
        cases=cases,
        seed=cfg["project"]["seed"],
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

    out_metrics_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    out_unc_dir = Path(cfg["paths"]["uncertainty_dir"]) / run_name
    out_fig_dir = Path(cfg["paths"]["figures_dir"]) / run_name
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_unc_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    num_classes = cfg["model"]["num_classes"]
    spacing = tuple(cfg["preprocessing"]["pixdim"])

    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], list) else str(batch["case_id"])
        image_tensor = batch["image"]
        image_np = image_tensor.detach().cpu().numpy()[0, 0]
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        probs_stack = mc_dropout_probs(
            model=model,
            image=image_tensor,
            cfg=cfg,
            device=device,
            num_samples=num_samples,
        )

        mean_probs = probs_stack.mean(axis=0)
        variance_map = probs_stack.var(axis=0).mean(axis=0)
        entropy_map = predictive_entropy(mean_probs)
        pred = mean_probs.argmax(axis=0).astype(np.int16)

        metrics = segmentation_metrics(
            pred_labels=pred,
            true_labels=label,
            num_classes=num_classes,
            spacing=spacing,
            include_background=False,
        )

        error_mask = pred != label
        foreground_mask = np.logical_or(label > 0, pred > 0)

        row = {
            "case_id": case_id,
            "split": args.split,
            "mc_samples": num_samples,
            "entropy_mean": float(np.mean(entropy_map)),
            "entropy_max": float(np.max(entropy_map)),
            "entropy_foreground_mean": float(np.mean(entropy_map[foreground_mask])) if foreground_mask.any() else np.nan,
            "entropy_error_mean": float(np.mean(entropy_map[error_mask])) if error_mask.any() else np.nan,
            "variance_mean": float(np.mean(variance_map)),
            "variance_max": float(np.max(variance_map)),
            "variance_foreground_mean": float(np.mean(variance_map[foreground_mask])) if foreground_mask.any() else np.nan,
            "variance_error_mean": float(np.mean(variance_map[error_mask])) if error_mask.any() else np.nan,
        }
        row.update(metrics)
        rows.append(row)

        np.save(out_unc_dir / f"{case_id}_entropy.npy", entropy_map.astype(np.float32))
        np.save(out_unc_dir / f"{case_id}_variance.npy", variance_map.astype(np.float32))
        np.save(out_unc_dir / f"{case_id}_prediction.npy", pred.astype(np.int16))

        if i < args.num_vis:
            save_uncertainty_figure(
                image=image_np,
                label=label,
                pred=pred,
                uncertainty=entropy_map,
                out_path=out_fig_dir / f"{case_id}_uncertainty.png",
                title=f"{case_id} | Dice={metrics['dice_mean']:.3f}",
            )

        print(
            f"{case_id}: Dice={metrics['dice_mean']:.4f}, "
            f"entropy_fg={row['entropy_foreground_mean']:.4f}"
        )

    df = pd.DataFrame(rows)
    out_csv = out_metrics_dir / f"{args.split}_uncertainty.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved uncertainty metrics: {out_csv}")
    print(f"Saved uncertainty maps: {out_unc_dir}")
    print(f"Saved figures: {out_fig_dir}")


if __name__ == "__main__":
    main()
