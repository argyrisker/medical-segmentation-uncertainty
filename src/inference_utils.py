from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference


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
    image = image.to(device)
    logits = sliding_window_inference(
        inputs=image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    probs = torch.softmax(logits, dim=1)
    if probs.shape[1] != num_classes:
        raise RuntimeError(f"Expected {num_classes} classes, got {probs.shape[1]}")
    return probs
