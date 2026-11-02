from __future__ import annotations

from typing import Any, Sequence

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)

CROP_MODES = ("after_norm", "before_norm", "none")
AUG_LEVELS = ("full", "anatomical", "none")


def _load_and_normalize(keys: list[str], pixdim: Sequence[float], crop_mode: str, crop_margin: int) -> list:
    """
    Shared deterministic preprocessing.

    crop_mode:
      after_norm  - v1 behaviour. CropForegroundd keeps voxels > 0, which after
                    z-scoring means above-mean voxels only. On MSD hippocampus
                    those touch every image edge, and since MONAI 1.5
                    (allow_smaller=False) the margin then pads instead of
                    cropping: every volume gets a 5-voxel zero border per side.
                    No labelled voxels are lost.
      before_norm - crop on raw intensities (> 0), then z-score. The raw scans
                    have almost no zero voxels, so this changes nothing here.
      none        - no crop. MSD hippocampus volumes are already cropped
                    (max 43 x 59 x 47 voxels).
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_foreground must be one of {CROP_MODES}, got {crop_mode}")

    modes = ("bilinear", "nearest") if len(keys) == 2 else ("bilinear",)
    crop = CropForegroundd(keys=keys, source_key="image", margin=crop_margin)
    norm = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)

    steps = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        # After RAS reorientation: axis 0 = left-right, axis 1 = posterior-anterior,
        # axis 2 = inferior-superior.
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(pixdim), mode=modes),
    ]
    if crop_mode == "before_norm":
        steps += [crop, norm]
    elif crop_mode == "after_norm":
        steps += [norm, crop]
    else:
        steps += [norm]
    return steps


def _augmentations(level: str, rot90: bool = True, patch_size: Sequence[int] | None = None) -> list:
    """
    full       - flips on all three axes and an intensity shift, plus (when
                 rot90 is true, as in v1) 90-degree rotations in the axial
                 plane. Flipping the anterior-posterior axis creates
                 anatomically impossible images, because the labels are
                 anterior (1) and posterior (2) hippocampus.
    anatomical - left-right flip only (hippocampi are mirror images) plus
                 intensity shift.
    none       - no augmentation (used for the overfitting study).
    """
    if level not in AUG_LEVELS:
        raise ValueError(f"augmentation.level must be one of {AUG_LEVELS}, got {level}")
    if level == "none":
        return []
    if level == "anatomical":
        return [
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        ]
    steps = [
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
    ]
    if rot90:
        # A 90-degree turn swaps axes 0 and 1, so it only keeps the patch
        # shape (and lets patches be batched) when those sides are equal.
        if patch_size is not None and int(patch_size[0]) != int(patch_size[1]):
            raise ValueError(
                f"augmentation.rot90 needs equal patch sides on axes 0 and 1, got {list(patch_size)}. "
                "Set augmentation.rot90: false."
            )
        steps.append(RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)))
    steps.append(RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5))
    return steps


def build_train_transforms(cfg: dict[str, Any]) -> Compose:
    pre = cfg["preprocessing"]
    spatial_size = tuple(pre["patch_size"])
    return Compose(
        _load_and_normalize(["image", "label"], pre["pixdim"], pre["crop_foreground"], pre["crop_margin"])
        + [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            # Foreground-biased patch sampling. When the patch is as large as
            # the padded volume this simply returns the whole volume.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=pre["pos"],
                neg=pre["neg"],
                num_samples=pre["num_samples"],
                image_key="image",
                image_threshold=0,
            ),
        ]
        + _augmentations(cfg["augmentation"]["level"], bool(cfg["augmentation"]["rot90"]), spatial_size)
        + [EnsureTyped(keys=["image", "label"])]
    )


def build_eval_transforms(cfg: dict[str, Any], has_label: bool = True) -> Compose:
    pre = cfg["preprocessing"]
    keys = ["image", "label"] if has_label else ["image"]
    return Compose(
        _load_and_normalize(keys, pre["pixdim"], pre["crop_foreground"], pre["crop_margin"])
        + [EnsureTyped(keys=keys)]
    )


# ---------------------------------------------------------------------------
# v1 function signatures, kept for the data-exploration notebook.
# They reproduce the original behaviour (crop after normalisation, full
# augmentation with 90-degree rotations).
# ---------------------------------------------------------------------------
def _v1_cfg(spatial_size, pixdim, num_samples, pos, neg, crop_margin) -> dict[str, Any]:
    return {
        "preprocessing": {
            "patch_size": list(spatial_size),
            "pixdim": list(pixdim),
            "num_samples": num_samples,
            "pos": pos,
            "neg": neg,
            "crop_margin": crop_margin,
            "crop_foreground": "after_norm",
        },
        "augmentation": {"level": "full", "rot90": True},
    }


def get_train_transforms(
    spatial_size: Sequence[int] = (32, 32, 32),
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    num_samples: int = 4,
    pos: float = 2.0,
    neg: float = 1.0,
    crop_margin: int = 5,
) -> Compose:
    return build_train_transforms(_v1_cfg(spatial_size, pixdim, num_samples, pos, neg, crop_margin))


def get_val_transforms(
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    crop_margin: int = 5,
) -> Compose:
    return build_eval_transforms(_v1_cfg((32, 32, 32), pixdim, 4, 2.0, 1.0, crop_margin))
