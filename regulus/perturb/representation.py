"""Perturbation input semantics.

Regulus does not derive deltas or baselines. The input matrix is consumed as
provided and ``representation`` records whether it contains precomputed
deltas or post-perturbation states.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from regulus.perturb.spec import REPRESENTATIONS, normalize_representation

INPUT_REPRESENTATIONS = REPRESENTATIONS


def validate_input_matrix(values: np.ndarray, representation: str, *, name: str) -> np.ndarray:
    normalize_representation(representation)
    matrix = np.asarray(values, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def apply_anchor_plus_delta(
    base_vector: Union[np.ndarray, list[float]],
    target_indices: list[int],
    deltas: list[float],
) -> np.ndarray:
    """Add user-specified CFO edits to an anchor vector."""
    out = np.asarray(base_vector, dtype=np.float32).copy()
    if len(target_indices) != len(deltas):
        raise ValueError("target_indices and deltas must have equal length")
    for index, delta in zip(target_indices, deltas):
        out[int(index)] += float(delta)
    return out
