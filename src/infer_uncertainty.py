from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from src.config_utils import config_from_checkpoint, get_device, set_global_seed
from src.datasets import get_splits, make_monai_dataset
from src.inference_utils import inference_kwargs, predict_logits
from src.metrics import ece_score, predictive_entropy, segmentation_metrics
from src.models import build_model, enable_mc_dropout
from src.transforms import build_eval_transforms
from src.visualization import save_uncertainty_figure

MODES = ("deterministic", "mc_dropout", "ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict with uncertainty. deterministic: one pass, softmax entropy. "
            "mc_dropout: T passes with dropout on. ensemble: one pass per checkpoint."
        )
    )
    parser.add_argument("--config", default=None, help="Only used if a checkpoint has no stored config.")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One checkpoint, or several for --mode ensemble.")
    parser.add_argument("--mode", choices=MODES, default="mc_dropout")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--run-name", default=None, help="Output folder name. Required for ensembles.")
    parser.add_argument("--num-samples", type=int, default=None, help="MC dropout passes (default: uncertainty.mc_samples).")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-vis", type=int, default=8)
    parser.add_argument("--max-voxels-per-case", type=int, default=50000)
    parser.add_argument("--save-maps", action="store_true", help="Save entropy/variance/prediction arrays as .npy")
    return parser.parse_args()


def load_members(paths: list[str], fallback_config, device):
    members = []
    for path in paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = config_from_checkpoint(ckpt, fallback_config)
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        members.append((model, cfg, ckpt))
    return members


@torch.no_grad()
def predict_stack(members, image, mode: str, num_samples: int, device) -> np.ndarray:
    """Returns softmax probabilities, shape T x C x H x W x D."""
    out = []
    if mode == "deterministic":
        model, cfg, _ = members[0]
        model.eval()
        out.append(torch.softmax(predict_logits(model, image, device=device, **inference_kwargs(cfg)), 1))
    elif mode == "mc_dropout":
        model, cfg, _ = members[0]
        model.eval()
        enable_mc_dropout(model)
        for _ in range(num_samples):
            out.append(torch.softmax(predict_logits(model, image, device=device, **inference_kwargs(cfg)), 1))
        model.eval()
    else:
        for model, cfg, _ in members:
            model.eval()
            out.append(torch.softmax(predict_logits(model, image, device=device, **inference_kwargs(cfg)), 1))
    return np.stack([p.detach().cpu().numpy()[0] for p in out], axis=0)


def _sample(arrays: list[np.ndarray], max_n: int, rng: np.random.Generator) -> list[np.ndarray]:
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    idx = rng.choice(n, size=max_n, replace=False)
    return [a[idx] for a in arrays]


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    if args.mode == "ensemble" and len(args.checkpoint) < 2:
        raise SystemExit("--mode ensemble needs at least two checkpoints.")
    if args.mode != "ensemble" and len(args.checkpoint) != 1:
        raise SystemExit(f"--mode {args.mode} takes exactly one checkpoint.")
    if args.mode == "ensemble" and not args.run_name:
        raise SystemExit("--mode ensemble needs --run-name for the output folder.")

    members = load_members(args.checkpoint, args.config, device)
    cfg = members[0][1]
    for _, other, _ in members[1:]:
        if other["preprocessing"] != cfg["preprocessing"] or other["split"] != cfg["split"]:
            raise SystemExit("Ensemble members must share preprocessing and split settings.")
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    if args.mode == "mc_dropout" and float(cfg["model"].get("dropout", 0.0)) == 0.0:
        print("Warning: model has dropout 0, so all MC dropout samples are identical.")

    num_samples = args.num_samples or int(cfg["uncertainty"]["mc_samples"])
    run_name = args.run_name or members[0][2].get("run_name") or Path(args.checkpoint[0]).parent.name

    _, val_cases, test_cases = get_splits(cfg, download=args.download)
    eval_cases = val_cases if args.split == "val" else test_cases
    ds = make_monai_dataset(eval_cases, build_eval_transforms(cfg), cache_rate=0.0)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    out_metrics_dir = Path(cfg["paths"]["metrics_dir"]) / run_name
    out_unc_dir = Path(cfg["paths"]["uncertainty_dir"]) / run_name / args.mode
    out_fig_dir = Path(cfg["paths"]["figures_dir"]) / run_name / args.mode
    for d in (out_metrics_dir, out_fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    num_classes = cfg["model"]["num_classes"]
    spacing = tuple(cfg["preprocessing"]["pixdim"])
    rows = []
    pooled_entropy, pooled_error, pooled_conf = [], [], []

    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"])
        image_tensor = batch["image"]
        image_np = image_tensor.detach().cpu().numpy()[0, 0]
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        stack = predict_stack(members, image_tensor, args.mode, num_samples, device)
        mean_probs = stack.mean(axis=0)
        variance_map = stack.var(axis=0).mean(axis=0)
        entropy_map = predictive_entropy(mean_probs)
        pred = mean_probs.argmax(axis=0).astype(np.int16)
        confidence = mean_probs.max(axis=0)

        metrics = segmentation_metrics(pred, label, num_classes, spacing=spacing, include_background=False)

        error_mask = pred != label
        # Every error voxel lies where the truth or the prediction is foreground,
        # so this region holds all errors without drowning them in easy background.
        region = np.logical_or(label > 0, pred > 0)
        correct_region = region & ~error_mask

        ent_r, err_r, conf_r = entropy_map[region], error_mask[region], confidence[region]
        case_auroc = roc_auc_score(err_r, ent_r) if 0 < err_r.sum() < err_r.size else np.nan

        e, er, c = _sample([ent_r, err_r, conf_r], args.max_voxels_per_case, rng)
        pooled_entropy.append(e)
        pooled_error.append(er)
        pooled_conf.append(c)

        row = {
            "case_id": case_id,
            "split": args.split,
            "mode": args.mode,
            "n_passes": stack.shape[0],
            "entropy_mean": float(entropy_map.mean()),
            "entropy_foreground_mean": float(ent_r.mean()) if region.any() else np.nan,
            "entropy_error_mean": float(entropy_map[error_mask].mean()) if error_mask.any() else np.nan,
            "entropy_correct_foreground_mean": float(entropy_map[correct_region].mean()) if correct_region.any() else np.nan,
            "variance_mean": float(variance_map.mean()),
            "variance_foreground_mean": float(variance_map[region].mean()) if region.any() else np.nan,
            "variance_error_mean": float(variance_map[error_mask].mean()) if error_mask.any() else np.nan,
            "error_auroc": float(case_auroc),
        }
        row.update(metrics)
        rows.append(row)

        if args.save_maps:
            out_unc_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_unc_dir / f"{case_id}_entropy.npy", entropy_map.astype(np.float32))
            np.save(out_unc_dir / f"{case_id}_variance.npy", variance_map.astype(np.float32))
            np.save(out_unc_dir / f"{case_id}_prediction.npy", pred)

        if i < args.num_vis:
            save_uncertainty_figure(
                image=image_np,
                label=label,
                pred=pred,
                uncertainty=entropy_map,
                out_path=out_fig_dir / f"{args.split}_{case_id}_uncertainty.png",
                title=f"{case_id} | {args.mode} | Dice={metrics['dice_mean']:.3f}",
            )

        print(f"{case_id}: Dice={metrics['dice_mean']:.4f}, error AUROC={case_auroc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_metrics_dir / f"{args.split}_uncertainty_{args.mode}.csv", index=False)

    ent = np.concatenate(pooled_entropy)
    err = np.concatenate(pooled_error)
    conf = np.concatenate(pooled_conf)
    pooled_auroc = float(roc_auc_score(err, ent))
    fpr, tpr, _ = roc_curve(err, ent)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 1000)).astype(int))
    pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep]}).to_csv(
        out_metrics_dir / f"{args.split}_roc_{args.mode}.csv", index=False
    )
    ece_fg, ece_rows = ece_score(conf, ~err, n_bins=cfg["calibration"]["n_bins"])
    pd.DataFrame(ece_rows).to_csv(out_metrics_dir / f"{args.split}_calibration_{args.mode}_foreground.csv", index=False)

    summary = {
        "run_name": run_name,
        "mode": args.mode,
        "split": args.split,
        "n_passes": int(df["n_passes"].iloc[0]),
        "dice_mean": float(df["dice_mean"].mean()),
        "dice_std": float(df["dice_mean"].std()),
        "hd95_mean": float(df["hd95_mean"].mean()),
        "error_auroc_pooled": pooled_auroc,
        "error_auroc_case_mean": float(df["error_auroc"].mean()),
        "ece_foreground": float(ece_fg),
        "entropy_error_mean": float(df["entropy_error_mean"].mean()),
        "entropy_correct_foreground_mean": float(df["entropy_correct_foreground_mean"].mean()),
        "dice_vs_entropy_pearson_r": float(df[["dice_mean", "entropy_foreground_mean"]].corr().iloc[0, 1]),
    }
    pd.DataFrame([summary]).to_csv(out_metrics_dir / f"{args.split}_uncertainty_{args.mode}_summary.csv", index=False)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {out_metrics_dir}")


if __name__ == "__main__":
    main()
