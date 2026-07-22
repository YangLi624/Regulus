"""Virtual CFO editing and before/after score comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from regulus.io.obs_filter import cohort_mean_sample, compute_ranks, resolve_dataset_indices
from regulus.manipulate.resolver import resolve_cfo_targets
from regulus.perturb.representation import apply_anchor_plus_delta
from regulus.perturb.spec import normalize_mode


@torch.no_grad()
def run_manipulation(
    regulus_model,
    *,
    cfo_targets: Sequence[str],
    cfo_delta: Sequence[float],
    anchor_h5ad: str | Path,
    obs_filter: Optional[str] = None,
    mode: str = "cfo_only",
    edit_mode: str = "anchor_plus_delta",
    top_k: int = 20,
    sort_by: str = "rank_gain",
) -> pd.DataFrame:
    """Apply a model-space CFO edit to a cohort-mean anchor."""
    if regulus_model.channels == "gene":
        raise ValueError("CFO manipulation requires a model trained with the CFO channel")
    if edit_mode != "anchor_plus_delta":
        raise ValueError("Only anchor_plus_delta is supported")
    mode = normalize_mode(mode, regulus_model.channels)
    dataset = regulus_model._dataset(anchor_h5ad)
    indices, n_cells = resolve_dataset_indices(dataset, obs_filter)
    sample = cohort_mean_sample(dataset, indices)
    cfo_base = sample.get("cfo_input")
    if cfo_base is None:
        raise KeyError("Anchor data do not contain a CFO activity matrix")
    target_indices, deltas = resolve_cfo_targets(
        cfo_targets, cfo_delta, regulus_model._hgt
    )
    edited = dict(sample)
    edited["cfo_input"] = torch.from_numpy(
        apply_anchor_plus_delta(cfo_base.numpy(), target_indices, deltas)
    )
    before = regulus_model._logits(sample, mode).squeeze(0).cpu().numpy()
    after = regulus_model._logits(edited, mode).squeeze(0).cpu().numpy()
    rank_before = compute_ranks(before)
    rank_after = compute_ranks(after)
    rank_gain = rank_before - rank_after
    score_delta = after - before
    order_values = {
        "rank_gain": rank_gain,
        "delta": score_delta,
        "after": after,
    }
    if sort_by not in order_values:
        raise ValueError("sort_by must be rank_gain, delta, or after")
    order = np.argsort(-order_values[sort_by])
    target_text = "|".join(map(str, cfo_targets))
    delta_text = "|".join(f"{value:.8g}" for value in deltas)
    rows = []
    for output_rank, candidate_index in enumerate(order[: min(top_k, len(order))], start=1):
        rows.append({
            "rank": output_rank,
            "perturbation": regulus_model.candidate_perturbations[int(candidate_index)],
            "rank_before": int(rank_before[candidate_index]),
            "rank_after": int(rank_after[candidate_index]),
            "rank_gain": int(rank_gain[candidate_index]),
            "score_before": float(before[candidate_index]),
            "score_after": float(after[candidate_index]),
            "score_delta": float(score_delta[candidate_index]),
            "cfo_targets": target_text,
            "cfo_delta": delta_text,
            "runtime_mode": mode,
            "edit_mode": edit_mode,
            "n_cells": n_cells,
            "obs_filter": obs_filter or "",
            "sort_by": sort_by,
            "n_candidates": len(regulus_model.candidate_perturbations),
        })
    return pd.DataFrame(rows)
