from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, list_data_collate
from monai.losses import DiceCELoss
from tqdm import tqdm

from src.config_utils import get_device, load_config, save_config, set_global_seed
from src.datasets import (
    MSDConfig,
    ensure_msd_downloaded,
    get_labelled_cases,
    make_monai_dataset,
    select_data_fraction,
    split_cases,
)
from src.inference_utils import predict_probabilities
from src.metrics import segmentation_metrics
from src.models import build_model
from src.transforms import get_train_transforms, get_val_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def validate(model, val_loader, cfg, device) -> float:
    num_classes = cfg["model"]["num_classes"]
    roi_size = tuple(cfg["inference"]["roi_size"])
    sw_batch_size = cfg["inference"]["sw_batch_size"]
    overlap = cfg["inference"]["overlap"]
    spacing = tuple(cfg["preprocessing"]["pixdim"])

    model.eval()
    dice_values = []

    for batch in val_loader:
        image = batch["image"].to(device)
        label = batch["label"].detach().cpu().numpy()[0, 0].astype(np.int16)

        probs = predict_probabilities(
            model=model,
            image=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            num_classes=num_classes,
            device=device,
            overlap=overlap,
        )

        pred = probs.argmax(dim=1).detach().cpu().numpy()[0].astype(np.int16)
        row = segmentation_metrics(
            pred_labels=pred,
            true_labels=label,
            num_classes=num_classes,
            spacing=spacing,
            include_background=False,
        )
        dice_values.append(row["dice_mean"])

    return float(np.nanmean(dice_values))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    base_run_name = args.run_name or cfg["project"]["run_name"]
    if args.data_fraction < 1.0:
        run_name = f"{base_run_name}_frac{args.data_fraction:g}"
    else:
        run_name = base_run_name

    run_dir = Path(cfg["paths"]["checkpoints_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_used.yaml")

    msd_cfg = MSDConfig(
        root_dir=cfg["dataset"]["root_dir"],
        task=cfg["dataset"]["task"],
        seed=seed,
    )
    ensure_msd_downloaded(msd_cfg, download=args.download)
    cases = get_labelled_cases(msd_cfg)

    train_cases, val_cases, test_cases = split_cases(
        cases=cases,
        seed=seed,
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
    )
    train_cases = select_data_fraction(train_cases, fraction=args.data_fraction, seed=seed)

    print(f"Total labelled cases: {len(cases)}")
    print(f"Train cases used: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    print(f"Internal test cases: {len(test_cases)}")

    train_tf = get_train_transforms(
        spatial_size=cfg["preprocessing"]["patch_size"],
        pixdim=cfg["preprocessing"]["pixdim"],
        num_samples=cfg["preprocessing"]["num_samples"],
        pos=cfg["preprocessing"]["pos"],
        neg=cfg["preprocessing"]["neg"],
        crop_margin=cfg["preprocessing"]["crop_margin"],
    )

    val_tf = get_val_transforms(
        pixdim=cfg["preprocessing"]["pixdim"],
        crop_margin=cfg["preprocessing"]["crop_margin"],
    )

    train_ds = make_monai_dataset(
        train_cases,
        train_tf,
        cache_rate=cfg["training"]["cache_rate"],
        num_workers=cfg["training"]["num_workers"],
    )
    val_ds = make_monai_dataset(
        val_cases,
        val_tf,
        cache_rate=cfg["training"]["cache_rate"],
        num_workers=cfg["training"]["num_workers"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)

    loss_fn = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    best_val_dice = -1.0
    best_epoch = -1
    history = []

    start = time.time()
    epochs = int(cfg["training"]["epochs"])
    val_interval = int(cfg["training"]["val_interval"])

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        mean_loss = float(np.mean(epoch_losses))

        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_dice": np.nan,
        }

        if epoch % val_interval == 0 or epoch == epochs:
            val_dice = validate(model, val_loader, cfg, device)
            row["val_dice"] = val_dice
            print(f"Epoch {epoch:03d} | train_loss={mean_loss:.4f} | val_dice={val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_dice": best_val_dice,
                    "config": cfg,
                    "run_name": run_name,
                    "data_fraction": args.data_fraction,
                }
                torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            print(f"Epoch {epoch:03d} | train_loss={mean_loss:.4f}")

        history.append(row)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "best_val_dice": best_val_dice,
        "config": cfg,
        "run_name": run_name,
        "data_fraction": args.data_fraction,
    }
    torch.save(final_checkpoint, run_dir / "last_model.pt")

    import pandas as pd

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    elapsed_min = (time.time() - start) / 60
    print(f"Training complete. Best val Dice={best_val_dice:.4f} at epoch {best_epoch}.")
    print(f"Elapsed: {elapsed_min:.1f} min")
    print(f"Best checkpoint: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
