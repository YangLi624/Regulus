"""Channel scheduling and classification metrics."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ENCODE_MODES: Tuple[str, ...] = ("gene_only", "cfo_only", "joint")


def normalize_mode_schedule(schedule: Dict[str, float]) -> Dict[str, float]:
    if not schedule:
        raise ValueError("channel_schedule cannot be empty")
    unknown = set(schedule) - set(ENCODE_MODES)
    if unknown:
        raise ValueError(f"unknown channel modes: {sorted(unknown)}")
    total = float(sum(schedule.values()))
    if total <= 0:
        raise ValueError("channel_schedule weights must sum to a positive value")
    return {key: float(value) / total for key, value in schedule.items()}


def sample_encode_mode(
    schedule: Dict[str, float],
    rng: random.Random | None = None,
) -> str:
    normalized = normalize_mode_schedule(schedule)
    chooser = rng or random
    return chooser.choices(list(normalized), weights=list(normalized.values()), k=1)[0]


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk_list: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    if logits.shape[0] == 0:
        return {f"top{k}_acc": 0.0 for k in topk_list}
    max_k = min(max(topk_list), logits.shape[1])
    top_indices = logits.topk(max_k, dim=1).indices
    return {
        f"top{k}_acc": float(
            (top_indices[:, : min(k, max_k)] == labels[:, None]).any(dim=1).float().mean()
        )
        for k in topk_list
    }


def aggregate_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metric_lists.items()}
