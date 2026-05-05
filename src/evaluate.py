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
from src.metrics import segmentation_metrics
from src.models import build_model
from src.transforms import get_val_transforms
from src.visualization import save_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-vis", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg["project"]["seed"]))
    device = get_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    run_name = args.run_name or checkpoint.get("run_name") or Path(args.checkpoint).parent.name

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
    ds = make_monai_dataset(
        eval_cases,
        tf,
        cache_rate=0.0,
        num_workers=cfg["training"]["num_workers"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    out_metrics_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    out_fig_dir = Path(cfg["paths"]["predictions_dir"]) / run_name
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    num_classes = cfg["model"]["num_classes"]
    spacing = tuple(cfg["preprocessing"]["pixdim"])

    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], list) else str(batch["case_id"])
        image_tensor = batch["image"]
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        probs = predict_probabilities(
            model=model,
            image=image_tensor,
            roi_size=tuple(cfg["inference"]["roi_size"]),
            sw_batch_size=cfg["inference"]["sw_batch_size"],
            num_classes=num_classes,
            device=device,
            overlap=cfg["inference"]["overlap"],
        )

        pred = probs.argmax(dim=1).detach().cpu().numpy()[0].astype(np.int16)

        metrics = segmentation_metrics(
            pred_labels=pred,
            true_labels=label,
            num_classes=num_classes,
            spacing=spacing,
            include_background=False,
        )

        row = {"case_id": case_id, "split": args.split}
        row.update(metrics)
        rows.append(row)

        if i < args.num_vis:
            image = image_tensor.detach().cpu().numpy()[0, 0]
            save_prediction_figure(
                image=image,
                label=label,
                pred=pred,
                out_path=out_fig_dir / f"{case_id}_prediction.png",
                title=f"{case_id} | Dice={metrics['dice_mean']:.3f}",
            )

        print(f"{case_id}: Dice={metrics['dice_mean']:.4f}, HD95={metrics['hd95_mean']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = out_metrics_dir / f"{args.split}_metrics.csv"
    df.to_csv(out_csv, index=False)

    summary = df.drop(columns=["case_id", "split"]).mean(numeric_only=True)
    summary.to_csv(out_metrics_dir / f"{args.split}_metrics_summary.csv")

    print("\nSummary:")
    print(summary)
    print(f"\nSaved metrics: {out_csv}")
    print(f"Saved figures: {out_fig_dir}")


if __name__ == "__main__":
    main()
