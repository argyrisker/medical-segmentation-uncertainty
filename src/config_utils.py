from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from monai.utils import set_determinism


# Defaults for keys added after the first version of this project.
# They reproduce the original (v1) behaviour, so old configs and old
# checkpoints keep working unchanged.
DEFAULTS: dict[str, Any] = {
    "split": {
        "seed": None,  # None -> use project.seed (original behaviour)
        "train_cases": None,  # int -> use only this many training cases
    },
    "preprocessing": {
        # after_norm (v1) | before_norm | none
        "crop_foreground": "after_norm",
    },
    "augmentation": {
        # full (v1): flips on all axes + rot90 + intensity shift
        # anatomical: left-right flip + intensity shift only
        # none: no augmentation
        "level": "full",
        "rot90": True,  # 90-degree axial rotations within level "full" (v1)
    },
    "model": {
        "arch": "monai",  # monai (v1) | custom
        "norm": "instance",  # instance | batch (custom arch only)
        "use_skips": True,  # custom arch only
    },
    "training": {
        "loss": "dicece",  # dicece | dice | ce
        "scheduler": "none",  # none | cosine
        "min_lr_ratio": 0.01,  # cosine floor as a fraction of lr
        "samples_per_epoch": None,  # int -> repeat small training sets
        "train_eval_cases": 20,  # fixed training subset for train Dice/loss
    },
}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `update` into a copy of `base`. Lists are replaced."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply command-line overrides such as `training.lr=0.001`."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = cfg
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg


def with_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill in keys that older configs do not have."""
    return deep_update(DEFAULTS, cfg)


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_cfg = _load_yaml_with_base((path.parent / base).resolve())
        cfg = deep_update(base_cfg, cfg)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """
    Load a YAML config.

    A config may contain `base: other.yaml` (relative to its own folder);
    it then only needs the keys that differ from the base.
    """
    cfg = _load_yaml_with_base(Path(path))
    cfg = apply_overrides(cfg, overrides)
    return with_defaults(cfg)


def config_from_checkpoint(checkpoint: dict[str, Any], fallback_path: str | Path | None = None) -> dict[str, Any]:
    """Use the exact config a model was trained with, so architecture always matches."""
    if checkpoint.get("config"):
        return with_defaults(checkpoint["config"])
    if fallback_path is None:
        raise ValueError("Checkpoint has no config and no fallback config was given.")
    return load_config(fallback_path)


def split_seed(cfg: dict[str, Any]) -> int:
    """Seed for the train/val/test split. Kept separate from the training seed,
    so models trained with different seeds share the same test set."""
    seed = cfg["split"].get("seed")
    return int(cfg["project"]["seed"] if seed is None else seed)


def resolve_run_name(
    cfg: dict[str, Any],
    run_name: str | None = None,
    data_fraction: float = 1.0,
    seed_override: int | None = None,
) -> str:
    name = run_name or cfg["project"]["run_name"]
    if data_fraction < 1.0:
        name = f"{name}_frac{data_fraction:g}"
    if seed_override is not None:
        name = f"{name}_seed{seed_override}"
    return name


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_determinism(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tuple3(value: Any) -> tuple[int | float, int | float, int | float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"Expected sequence of length 3, got: {value}")


def make_run_dir(base_dir: str | Path, run_name: str) -> Path:
    path = Path(base_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
