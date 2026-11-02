from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, list_data_collate
from tqdm import tqdm

from src.config_utils import (
    get_device,
    load_config,
    resolve_run_name,
    save_config,
    save_json,
    set_global_seed,
)
from src.datasets import get_splits, make_monai_dataset, repeat_to_length
from src.inference_utils import inference_kwargs, predict_logits
from src.losses import build_loss
from src.metrics import mean_foreground_dice
from src.models import build_model, count_parameters
from src.transforms import build_eval_transforms, build_train_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model.")
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--run-name", default=None, help="Defaults to project.run_name in the config.")
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed. The data split stays fixed. Adds _seed<N> to the run name.",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.lr=0.001 model.dropout=0.0",
    )
    return parser.parse_args()


def _tensor(x):
    """Strip MONAI metadata for faster tensor maths."""
    return x.as_tensor() if hasattr(x, "as_tensor") else x


@torch.no_grad()
def evaluate_volumes(model, loader, loss_fn, cfg, device) -> tuple[float, float, list[dict]]:
    """
    Whole-volume evaluation in eval mode: sliding-window logits, the training
    loss on those logits, and mean foreground Dice.
    """
    num_classes = cfg["model"]["num_classes"]
    model.eval()
    losses, dices, per_case = [], [], []

    for batch in loader:
        image = _tensor(batch["image"]).to(device)
        label_t = _tensor(batch["label"]).to(device)
        logits = predict_logits(model, image, device=device, **inference_kwargs(cfg))

        losses.append(float(loss_fn(logits, label_t).item()))
        pred = logits.argmax(dim=1).cpu().numpy()[0].astype(np.int16)
        label = label_t.cpu().numpy()[0, 0].astype(np.int16)
        dice = mean_foreground_dice(pred, label, num_classes)
        dices.append(dice)

        case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"])
        per_case.append({"case_id": case_id, "dice_mean": dice})

    return float(np.mean(losses)), float(np.nanmean(dices)), per_case


def build_scheduler(optimizer, cfg, epochs: int):
    name = cfg["training"].get("scheduler", "none")
    if name == "none":
        return None
    if name == "cosine":
        lr = float(cfg["training"]["lr"])
        eta_min = lr * float(cfg["training"].get("min_lr_ratio", 0.01))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    raise ValueError(f"Unknown training.scheduler: {name}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    # Freeze the split seed before any training-seed override, so every
    # seed of the same experiment sees the same train/val/test cases.
    if cfg["split"].get("seed") is None:
        cfg["split"]["seed"] = int(cfg["project"]["seed"])
    if args.seed is not None:
        cfg["project"]["seed"] = int(args.seed)

    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    run_name = resolve_run_name(cfg, args.run_name, args.data_fraction, args.seed)
    cfg["project"]["run_name"] = run_name
    run_dir = Path(cfg["paths"]["checkpoints_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_used.yaml")

    train_cases, val_cases, test_cases = get_splits(cfg, download=args.download, data_fraction=args.data_fraction)
    n_eval = int(cfg["training"].get("train_eval_cases", 20))
    train_eval_cases = train_cases[: min(n_eval, len(train_cases))]
    train_list = repeat_to_length(train_cases, cfg["training"].get("samples_per_epoch"))

    print(f"Run: {run_name}")
    print(f"Train cases used: {len(train_cases)} (samples per epoch: {len(train_list)})")
    print(f"Val cases: {len(val_cases)} | Internal test cases: {len(test_cases)}")
    print(f"Train-eval subset for train Dice/loss: {len(train_eval_cases)} cases")

    tcfg = cfg["training"]
    num_workers = int(tcfg["num_workers"])
    train_ds = make_monai_dataset(train_list, build_train_transforms(cfg), tcfg["cache_rate"], num_workers)
    eval_tf = build_eval_transforms(cfg)
    val_ds = make_monai_dataset(val_cases, eval_tf, tcfg["cache_rate"], num_workers)
    train_eval_ds = make_monai_dataset(train_eval_cases, eval_tf, tcfg["cache_rate"], num_workers)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model: {cfg['model']['arch']} | trainable parameters: {n_params:,}")

    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    epochs = int(tcfg["epochs"])
    scheduler = build_scheduler(optimizer, cfg, epochs)
    val_interval = int(tcfg["val_interval"])

    best_val_dice = -math.inf
    best_epoch = -1
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        lr_now = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = _tensor(batch["image"]).to(device)
            labels = _tensor(batch["label"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr_now,
            # Mean loss over augmented training patches, model in train mode
            # (dropout on). Not directly comparable with val_loss.
            "train_loss": float(np.mean(epoch_losses)),
            # Same whole-volume, eval-mode procedure as validation, on a fixed
            # training subset. Use these against val_loss / val_dice.
            "train_eval_loss": np.nan,
            "train_eval_dice": np.nan,
            "val_loss": np.nan,
            "val_dice": np.nan,
        }

        if epoch % val_interval == 0 or epoch == epochs:
            val_loss, val_dice, val_per_case = evaluate_volumes(model, val_loader, loss_fn, cfg, device)
            tr_loss, tr_dice, _ = evaluate_volumes(model, train_eval_loader, loss_fn, cfg, device)
            row.update(train_eval_loss=tr_loss, train_eval_dice=tr_dice, val_loss=val_loss, val_dice=val_dice)

            improved = val_dice > best_val_dice
            if improved:
                best_val_dice = val_dice
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_val_dice": best_val_dice,
                        "config": cfg,
                        "run_name": run_name,
                        "data_fraction": args.data_fraction,
                    },
                    run_dir / "best_model.pt",
                )
                pd.DataFrame(val_per_case).to_csv(run_dir / "val_per_case_best.csv", index=False)

            print(
                f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={row['train_loss']:.4f} | "
                f"train_eval_loss={tr_loss:.4f} train_eval_dice={tr_dice:.4f} | "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}" + (" *" if improved else "")
            )
        else:
            print(f"Epoch {epoch:03d} | lr={lr_now:.2e} | train_loss={row['train_loss']:.4f}")

        row["epoch_time_s"] = time.time() - epoch_start
        history.append(row)
        # Written every epoch, so curves can be plotted while training runs.
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "best_val_dice": best_val_dice,
            "config": cfg,
            "run_name": run_name,
            "data_fraction": args.data_fraction,
        },
        run_dir / "last_model.pt",
    )

    elapsed_min = (time.time() - start) / 60
    last = history[-1]
    save_json(
        {
            "run_name": run_name,
            "config_file": str(args.config),
            "overrides": list(args.set),
            "seed": seed,
            "split_seed": int(cfg["split"]["seed"]),
            "data_fraction": args.data_fraction,
            "n_train_cases": len(train_cases),
            "n_val_cases": len(val_cases),
            "n_test_cases": len(test_cases),
            "trainable_parameters": n_params,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "final_train_eval_dice": last["train_eval_dice"],
            "final_val_dice": last["val_dice"],
            "final_train_eval_loss": last["train_eval_loss"],
            "final_val_loss": last["val_loss"],
            "elapsed_min": elapsed_min,
            "device": str(device),
        },
        run_dir / "summary.json",
    )

    print(f"Training complete. Best val Dice={best_val_dice:.4f} at epoch {best_epoch}.")
    print(f"Elapsed: {elapsed_min:.1f} min")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
