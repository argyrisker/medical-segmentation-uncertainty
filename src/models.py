from __future__ import annotations

import torch
from monai.networks.nets import UNet


def get_unet(
    in_channels: int = 1,
    num_classes: int = 3,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm="INSTANCE",
        dropout=dropout,
    )


def build_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return get_unet(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_classes", 3),
        channels=tuple(model_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(model_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=model_cfg.get("num_res_units", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Keep the model in eval mode, but reactivate dropout layers for MC dropout.
    """
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
            ),
        ):
            module.train()
