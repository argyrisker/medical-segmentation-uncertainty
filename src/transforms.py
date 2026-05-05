from __future__ import annotations

from typing import Sequence

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


def get_train_transforms(
    spatial_size: Sequence[int] = (32, 32, 32),
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    num_samples: int = 4,
    pos: float = 2.0,
    neg: float = 1.0,
    crop_margin: int = 5,
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(pixdim),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=crop_margin),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(spatial_size)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(spatial_size),
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    crop_margin: int = 5,
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(pixdim),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image", margin=crop_margin),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_inference_transforms(
    pixdim: Sequence[float] = (1.0, 1.0, 1.0),
    crop_margin: int = 5,
    has_label: bool = True,
) -> Compose:
    keys = ["image", "label"] if has_label else ["image"]
    modes = ("bilinear", "nearest") if has_label else ("bilinear",)

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=tuple(pixdim), mode=modes),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=keys, source_key="image", margin=crop_margin),
            EnsureTyped(keys=keys),
        ]
    )
