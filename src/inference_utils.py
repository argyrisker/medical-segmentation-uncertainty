from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    device: torch.device,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Sliding-window logits. The caller decides train/eval mode of the model."""
    image = image.to(device)
    return sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    image: torch.Tensor,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    num_classes: int,
    device: torch.device,
    overlap: float = 0.5,
) -> torch.Tensor:
    model.eval()
    logits = predict_logits(model, image, roi_size, sw_batch_size, device, overlap)
    probs = torch.softmax(logits, dim=1)
    if probs.shape[1] != num_classes:
        raise RuntimeError(f"Expected {num_classes} classes, got {probs.shape[1]}")
    return probs


def inference_kwargs(cfg: dict) -> dict:
    inf = cfg["inference"]
    return {
        "roi_size": tuple(inf["roi_size"]),
        "sw_batch_size": inf["sw_batch_size"],
        "overlap": inf["overlap"],
    }
