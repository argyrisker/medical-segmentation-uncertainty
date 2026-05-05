from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from monai.apps import DecathlonDataset
from monai.data import CacheDataset, Dataset


@dataclass(frozen=True)
class MSDConfig:
    root_dir: str | Path = "data"
    task: str = "Task04_Hippocampus"
    seed: int = 42

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir)

    @property
    def task_path(self) -> Path:
        return self.root_path / self.task

    @property
    def dataset_json_path(self) -> Path:
        return self.task_path / "dataset.json"


def ensure_msd_downloaded(cfg: MSDConfig, download: bool = True) -> None:
    cfg.root_path.mkdir(parents=True, exist_ok=True)
    _ = DecathlonDataset(
        root_dir=str(cfg.root_path),
        task=cfg.task,
        section="training",
        transform=None,
        download=download,
        seed=cfg.seed,
        val_frac=0.2,
        cache_num=0,
        num_workers=0,
    )


def load_dataset_json(cfg: MSDConfig) -> dict[str, Any]:
    if not cfg.dataset_json_path.exists():
        raise FileNotFoundError(
            f"Could not find {cfg.dataset_json_path}. "
            "Run with download=True first."
        )
    with open(cfg.dataset_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_msd_path(cfg: MSDConfig, path_value: str) -> Path:
    return cfg.task_path / path_value.replace("./", "")


def get_labelled_cases(cfg: MSDConfig) -> list[dict[str, Any]]:
    info = load_dataset_json(cfg)
    cases = []
    for item in info["training"]:
        image_path = resolve_msd_path(cfg, item["image"])
        label_path = resolve_msd_path(cfg, item["label"])
        case_id = image_path.name.replace(".nii.gz", "")
        cases.append(
            {
                "case_id": case_id,
                "image": str(image_path),
                "label": str(label_path),
            }
        )
    return cases


def split_cases(
    cases: list[dict[str, Any]],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be in [0, 1).")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.")

    rng = random.Random(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_frac))
    n_val = int(round(n_total * val_frac))

    test_cases = shuffled[:n_test]
    val_cases = shuffled[n_test:n_test + n_val]
    train_cases = shuffled[n_test + n_val:]

    return train_cases, val_cases, test_cases


def select_data_fraction(
    train_cases: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0:
        return train_cases

    rng = random.Random(seed)
    selected = train_cases.copy()
    rng.shuffle(selected)
    n = max(1, int(round(len(selected) * fraction)))
    return selected[:n]


def make_monai_dataset(
    data: list[dict[str, Any]],
    transform,
    cache_rate: float = 0.0,
    num_workers: int = 0,
):
    if cache_rate and cache_rate > 0:
        return CacheDataset(
            data=data,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    return Dataset(data=data, transform=transform)


def load_nifti(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    return np.asarray(nii.dataobj), nii


def case_basic_stats(case: dict[str, Any]) -> dict[str, Any]:
    image, image_nii = load_nifti(case["image"])
    label, _ = load_nifti(case["label"])
    label_values, label_counts = np.unique(label, return_counts=True)
    spacing = image_nii.header.get_zooms()[:3]

    return {
        "case_id": case["case_id"],
        "image_shape": tuple(int(x) for x in image.shape),
        "label_shape": tuple(int(x) for x in label.shape),
        "spacing": tuple(float(x) for x in spacing),
        "label_values": [int(x) for x in label_values],
        "label_counts": {int(v): int(c) for v, c in zip(label_values, label_counts)},
        "foreground_voxels": int((label > 0).sum()),
        "foreground_fraction": float((label > 0).mean()),
        "image_p01": float(np.percentile(image, 1)),
        "image_p50": float(np.percentile(image, 50)),
        "image_p99": float(np.percentile(image, 99)),
    }
