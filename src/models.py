from __future__ import annotations

from typing import Any, Sequence

import torch
from monai.networks.nets import UNet
from torch import nn


# ---------------------------------------------------------------------------
# v1 model: MONAI UNet (always has skip connections)
# ---------------------------------------------------------------------------
def get_unet(
    in_channels: int = 1,
    num_classes: int = 3,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.1,
    norm: str = "INSTANCE",
) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Custom 3D U-Net with switches for the ablation study
# ---------------------------------------------------------------------------
def make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"norm must be 'instance' or 'batch', got {norm}")


class ConvBlock(nn.Module):
    """
    Two 3x3x3 convolutions, each followed by normalisation, LeakyReLU and dropout.

    With residual=True the block input is added to the output of the second
    convolution (through a 1x1x1 convolution when channel counts differ),
    as in a ResNet unit.
    """

    def __init__(self, in_ch: int, out_ch: int, norm: str, dropout: float, residual: bool):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(norm, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_ch)
        self.act = nn.LeakyReLU(0.01)
        # nn.Dropout (element-wise) as in MONAI's UNet; MC dropout re-enables it.
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if residual:
            self.shortcut = (
                nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.drop(self.act(out))


class UNet3D(nn.Module):
    """
    Classic 3D U-Net.

    Encoder: one ConvBlock per level, 2x2x2 max pooling between levels.
    Decoder: 2x2x2 transposed convolution, concatenation with the encoder
    feature map of the same resolution (the skip connection), ConvBlock.
    Head: 1x1x1 convolution to class logits.

    `use_skips=False` removes every skip connection, so the decoder must
    rebuild spatial detail from the bottleneck alone.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        residual: bool = True,
        dropout: float = 0.1,
        norm: str = "instance",
        use_skips: bool = True,
    ):
        super().__init__()
        channels = list(channels)
        if len(channels) < 2:
            raise ValueError("Need at least two levels in model.channels.")
        self.use_skips = use_skips
        self.depth = len(channels)

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, norm, dropout, residual))
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 2, -1, -1):
            self.upsamplers.append(nn.ConvTranspose3d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            dec_in = channels[i] * 2 if use_skips else channels[i]
            self.decoders.append(ConvBlock(dec_in, channels[i], norm, dropout, residual))

        self.head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    @property
    def size_multiple(self) -> int:
        """Input sizes must be divisible by this number."""
        return 2 ** (self.depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.size_multiple
        if any(s % m for s in x.shape[2:]):
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} must be divisible by {m}. "
                "Set preprocessing.patch_size and inference.roi_size accordingly."
            )
        features = []
        for i, enc in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = enc(x)
            features.append(x)

        for j, (up, dec) in enumerate(zip(self.upsamplers, self.decoders)):
            x = up(x)
            if self.use_skips:
                x = torch.cat([x, features[-(j + 2)]], dim=1)
            x = dec(x)
        return self.head(x)


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    arch = model_cfg.get("arch", "monai")
    norm = model_cfg.get("norm", "instance")
    channels = tuple(model_cfg.get("channels", [16, 32, 64, 128, 256]))

    if arch == "monai":
        if not model_cfg.get("use_skips", True):
            raise ValueError("MONAI UNet always has skip connections. Use model.arch: custom.")
        return get_unet(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 3),
            channels=channels,
            strides=tuple(model_cfg.get("strides", [2] * (len(channels) - 1))),
            num_res_units=model_cfg.get("num_res_units", 2),
            dropout=model_cfg.get("dropout", 0.1),
            norm=norm.upper(),
        )

    if arch == "custom":
        strides = model_cfg.get("strides", [2] * (len(channels) - 1))
        if len(strides) != len(channels) - 1 or any(int(s) != 2 for s in strides):
            raise ValueError("Custom UNet3D downsamples by 2 between levels: strides must be all 2s.")
        model = UNet3D(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 3),
            channels=channels,
            residual=int(model_cfg.get("num_res_units", 2)) > 0,
            dropout=model_cfg.get("dropout", 0.1),
            norm=norm,
            use_skips=bool(model_cfg.get("use_skips", True)),
        )
        m = model.size_multiple
        for key, size in (
            ("preprocessing.patch_size", cfg["preprocessing"]["patch_size"]),
            ("inference.roi_size", cfg["inference"]["roi_size"]),
        ):
            if any(int(s) % m for s in size):
                raise ValueError(f"{key}={size} must be divisible by {m} for a {len(channels)}-level UNet3D.")
        return model

    raise ValueError(f"Unknown model.arch: {arch}")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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
