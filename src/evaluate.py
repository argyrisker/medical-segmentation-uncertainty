from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader

from src.config_utils import config_from_checkpoint, get_device, set_global_seed
from src.datasets import get_splits, make_monai_dataset
from src.inference_utils import inference_kwargs, predict_probabilities
from src.metrics import confusion_matrix, scores_from_confusion, segmentation_metrics
from src.models import build_model
from src.transforms import build_eval_transforms
from src.visualization import class_names, save_confusion_matrix, save_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on one split.")
    parser.add_argument("--config", default=None, help="Only used if the checkpoint has no stored config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-vis", type=int, default=8)
    return parser.parse_args()


def select_split(cfg, split: str, download: bool, data_fraction: float):
    train_cases, val_cases, test_cases = get_splits(cfg, download=download, data_fraction=data_fraction)
    return {"train": train_cases, "val": val_cases, "test": test_cases}[split]


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = config_from_checkpoint(checkpoint, args.config)
    set_global_seed(int(cfg["project"]["seed"]))
    run_name = args.run_name or checkpoint.get("run_name") or Path(args.checkpoint).parent.name

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_cases = select_split(cfg, args.split, args.download, float(checkpoint.get("data_fraction", 1.0)))
    ds = make_monai_dataset(eval_cases, build_eval_transforms(cfg), cache_rate=0.0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    out_metrics_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    out_fig_dir = Path(cfg["paths"]["predictions_dir"]) / run_name
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    num_classes = cfg["model"]["num_classes"]
    spacing = tuple(cfg["preprocessing"]["pixdim"])
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"])
        image_tensor = batch["image"]
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        probs = predict_probabilities(
            model=model,
            image=image_tensor,
            num_classes=num_classes,
            device=device,
            **inference_kwargs(cfg),
        )
        pred = probs.argmax(dim=1).detach().cpu().numpy()[0].astype(np.int16)

        metrics = segmentation_metrics(
            pred_labels=pred,
            true_labels=label,
            num_classes=num_classes,
            spacing=spacing,
            include_background=False,
        )
        cm_total += confusion_matrix(pred, label, num_classes)

        row = {"case_id": case_id, "split": args.split}
        row.update(metrics)
        rows.append(row)

        if i < args.num_vis:
            image = image_tensor.detach().cpu().numpy()[0, 0]
            save_prediction_figure(
                image=image,
                label=label,
                pred=pred,
                out_path=out_fig_dir / f"{args.split}_{case_id}_prediction.png",
                title=f"{case_id} | Dice={metrics['dice_mean']:.3f}",
            )

        print(f"{case_id}: Dice={metrics['dice_mean']:.4f}, HD95={metrics['hd95_mean']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = out_metrics_dir / f"{args.split}_metrics.csv"
    df.to_csv(out_csv, index=False)

    numeric = df.drop(columns=["case_id", "split"])
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(), "median": numeric.median()})
    summary.to_csv(out_metrics_dir / f"{args.split}_metrics_summary.csv")

    # Voxel confusion matrix pooled over all cases (rows = truth, columns = prediction).
    names = class_names(num_classes)
    pd.DataFrame(cm_total, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(
        out_metrics_dir / f"{args.split}_confusion_matrix.csv"
    )
    pd.DataFrame([scores_from_confusion(cm_total)]).to_csv(out_metrics_dir / f"{args.split}_voxel_scores.csv", index=False)
    save_confusion_matrix(
        cm_total,
        names,
        out_metrics_dir / f"{args.split}_confusion_matrix.png",
        title=f"{run_name} | {args.split} | voxel confusion matrix",
    )

    print("\nSummary (mean / std over cases):")
    print(summary[["mean", "std"]].loc[["dice_mean", "hd95_mean", "precision_mean", "sensitivity_mean"]])
    print(f"\nSaved metrics: {out_csv}")
    print(f"Saved confusion matrix: {out_metrics_dir / f'{args.split}_confusion_matrix.png'}")
    print(f"Saved figures: {out_fig_dir}")


if __name__ == "__main__":
    main()
