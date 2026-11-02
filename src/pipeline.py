from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config_utils import load_config, resolve_run_name


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train -> evaluate -> uncertainty -> calibration -> analysis")
    parser.add_argument("--config", default="configs/hippocampus_unet.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--strong", action="store_true", help="Run low-data experiments too.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run very few epochs for smoke testing.")
    parser.add_argument("--train-only", action="store_true", help="Skip all evaluation steps.")
    return parser.parse_args()


def one_run(args, cfg, data_fraction: float, epochs_override: int | None = None) -> None:
    run_name = resolve_run_name(cfg, args.run_name, data_fraction, args.seed)
    checkpoint = Path(cfg["paths"]["checkpoints_dir"]) / run_name / "best_model.pt"

    train_cmd = [
        sys.executable, "-m", "src.train",
        "--config", args.config,
        "--data-fraction", str(data_fraction),
        "--device", args.device,
    ]
    if args.run_name:
        train_cmd += ["--run-name", args.run_name]
    if args.seed is not None:
        train_cmd += ["--seed", str(args.seed)]
    if args.set:
        train_cmd += ["--set", *args.set]
    if args.download:
        train_cmd.append("--download")
    if epochs_override is not None:
        train_cmd += ["--epochs", str(epochs_override)]
    run(train_cmd)

    if args.train_only:
        return

    common = ["--checkpoint", str(checkpoint), "--split", args.split, "--run-name", run_name, "--device", args.device]
    run([sys.executable, "-m", "src.evaluate", *common])
    run([sys.executable, "-m", "src.infer_uncertainty", *common, "--mode", "mc_dropout"])
    run([sys.executable, "-m", "src.calibration", *common])
    run([
        sys.executable, "-m", "src.analysis",
        "--config", args.config, "--run-name", run_name, "--split", args.split, "--mode", "mc_dropout",
    ])
    run([sys.executable, "-m", "src.plot_curves", "--runs", run_name])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    epochs_override = 2 if args.fast_dev_run else None
    fractions = cfg["experiments"]["low_data_fractions"] if args.strong else [1.0]

    for frac in fractions:
        one_run(args=args, cfg=cfg, data_fraction=float(frac), epochs_override=epochs_override)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
