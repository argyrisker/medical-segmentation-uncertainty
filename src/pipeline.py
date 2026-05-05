from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config_utils import load_config


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--strong", action="store_true", help="Run low-data experiments too.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run very few epochs for smoke testing.")
    return parser.parse_args()


def one_run(args, cfg, run_name: str, data_fraction: float, epochs_override: int | None = None) -> None:
    checkpoint = Path(cfg["paths"]["checkpoints_dir"]) / (
        f"{run_name}_frac{data_fraction:g}" if data_fraction < 1.0 else run_name
    ) / "best_model.pt"

    train_cmd = [
        sys.executable, "-m", "src.train",
        "--config", args.config,
        "--run-name", run_name,
        "--data-fraction", str(data_fraction),
        "--device", args.device,
    ]
    if args.download:
        train_cmd.append("--download")
    if epochs_override is not None:
        train_cmd.extend(["--epochs", str(epochs_override)])

    run(train_cmd)

    actual_run_name = checkpoint.parent.name

    eval_cmd = [
        sys.executable, "-m", "src.evaluate",
        "--config", args.config,
        "--checkpoint", str(checkpoint),
        "--split", args.split,
        "--run-name", actual_run_name,
        "--device", args.device,
    ]
    run(eval_cmd)

    unc_cmd = [
        sys.executable, "-m", "src.infer_uncertainty",
        "--config", args.config,
        "--checkpoint", str(checkpoint),
        "--split", args.split,
        "--run-name", actual_run_name,
        "--device", args.device,
    ]
    run(unc_cmd)

    cal_cmd = [
        sys.executable, "-m", "src.calibration",
        "--config", args.config,
        "--checkpoint", str(checkpoint),
        "--split", args.split,
        "--run-name", actual_run_name,
        "--device", args.device,
    ]
    run(cal_cmd)

    analysis_cmd = [
        sys.executable, "-m", "src.analysis",
        "--config", args.config,
        "--run-name", actual_run_name,
        "--split", args.split,
    ]
    run(analysis_cmd)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or cfg["project"]["run_name"]
    epochs_override = 2 if args.fast_dev_run else None

    if args.strong:
        fractions = cfg["experiments"]["low_data_fractions"]
    else:
        fractions = [1.0]

    for frac in fractions:
        one_run(
            args=args,
            cfg=cfg,
            run_name=run_name,
            data_fraction=float(frac),
            epochs_override=epochs_override,
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
