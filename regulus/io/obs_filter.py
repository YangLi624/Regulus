"""Cell filtering and cohort aggregation for inference workflows."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from regulus.perturb.data import PerturbationDataset


class PerturbationRowSubset(Dataset):
    """A non-empty row subset of a perturbation dataset."""

    def __init__(self, base: Dataset, indices: Sequence[int]):
        if not indices:
            raise ValueError("A perturbation subset must contain at least one row")
        invalid = [index for index in indices if index < 0 or index >= len(base)]
        if invalid:
            raise IndexError(f"Dataset row indices are out of range: {invalid[:5]}")
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return self.base[self.indices[index]]


def _perturb_obs_frame(dataset: PerturbationDataset) -> pd.DataFrame:
    hca_key = dataset.hca[0]
    perturb_rows = dataset.perturb_rows[hca_key]
    hca = dataset.reader.data[hca_key]["hca"]
    obs = hca.obs.iloc[perturb_rows].copy()
    obs["cell_id"] = obs.index.astype(str)
    obs["dataset_index"] = range(len(perturb_rows))
    return obs


def resolve_dataset_indices(
    dataset: PerturbationDataset,
    obs_filter: Optional[str] = None,
) -> tuple[list[int], int]:
    """Resolve a pandas query over H5AD ``obs`` to model-dataset rows."""
    row_count = len(dataset)
    if obs_filter is None or not str(obs_filter).strip():
        return list(range(row_count)), row_count
    obs = _perturb_obs_frame(dataset)
    try:
        matched = obs.query(str(obs_filter), engine="python")
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate obs_filter {obs_filter!r}; available fields include "
            f"H5AD obs columns, cell_id, and dataset_index: {exc}"
        ) from exc
    if matched.empty:
        raise ValueError(f"obs_filter {obs_filter!r} matched no cells")
    indices = matched["dataset_index"].astype(int).tolist()
    return indices, len(indices)


def subset_dataset(
    dataset: PerturbationDataset,
    obs_filter: Optional[str] = None,
) -> tuple[Dataset, int, list[int]]:
    indices, cell_count = resolve_dataset_indices(dataset, obs_filter)
    if cell_count == len(dataset):
        return dataset, cell_count, indices
    return PerturbationRowSubset(dataset, indices), cell_count, indices


def cohort_mean_sample(dataset: Dataset, indices: Sequence[int]) -> dict:
    """Average prepared gene and CFO inputs across selected cells."""
    if not indices:
        raise ValueError("A cohort mean requires at least one cell")
    samples = [dataset[int(index)] for index in indices]
    mean_sample: dict = {}
    for key in ("gene_input", "cfo_input"):
        tensors = [sample[key] for sample in samples if sample.get(key) is not None]
        if tensors:
            mean_sample[key] = torch.stack(tensors).mean(dim=0)
    reference = samples[0]
    mean_sample["cell_id"] = f"cohort_mean_n{len(indices)}"
    mean_sample["perturb_label"] = reference.get("perturb_label", "")
    mean_sample["perturb_type"] = reference.get("perturb_type", "")
    for key in ("gene_symbols", "representation"):
        if key in reference:
            mean_sample[key] = reference[key]
    post_expression = [sample["post_expr"] for sample in samples if "post_expr" in sample]
    if post_expression:
        mean_sample["post_expr"] = torch.stack(post_expression).mean(dim=0)
    return mean_sample


def compute_ranks(scores: np.ndarray) -> np.ndarray:
    """Return 1-based descending ranks with minimum-rank tie handling."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.ones(len(values), dtype=np.int64)
    for index in range(1, len(values)):
        sorted_ranks[index] = (
            sorted_ranks[index - 1]
            if sorted_values[index] == sorted_values[index - 1]
            else index + 1
        )
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = sorted_ranks
    return ranks
