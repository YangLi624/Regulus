"""Version-tolerant automatic mixed-precision helpers."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device | str, enabled: bool = True):
    device_type = torch.device(device).type
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type)
    if device_type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()
