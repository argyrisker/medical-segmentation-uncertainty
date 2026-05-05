from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.utils import set_determinism


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
