"""Unit tests for observation filters and rank-gain ordering."""

from pathlib import Path

import numpy as np
import pytest
import torch

from regulus.io.obs_filter import (
    PerturbationRowSubset,
    cohort_mean_sample,
    compute_ranks,
    resolve_dataset_indices,
)

ROOT = Path(__file__).parent.parent
H5AD = ROOT / "tasks/04_perturb/data/Norman_tf_single_random/Norman_tf_single_test_random.h5ad"
GENE_UNIVERSE = ROOT / "data/gene_universe.csv"


def test_compute_ranks_ordering():
    scores = np.array([0.1, 0.9, 0.5, 0.9])
    ranks = compute_ranks(scores)
    assert ranks[1] == 1
    assert ranks[3] == 1
    assert ranks[0] == 4
    assert ranks[2] == 3


def test_rank_gain_semantics():
    before = np.array([10.0, 5.0, 1.0])
    after = np.array([8.0, 6.0, 20.0])
    rb = compute_ranks(before)
    ra = compute_ranks(after)
    gain = rb - ra
    assert gain[2] == 2
    assert gain[0] == -1


def test_cohort_mean_sample_tensors():
    class _FakeDataset:
        def __init__(self):
            self._rows = [
                {
                    "gene_input": torch.tensor([1.0, 2.0]),
                    "cfo_input": torch.tensor([3.0, 4.0]),
                    "perturb_label": "A",
                },
                {
                    "gene_input": torch.tensor([3.0, 4.0]),
                    "cfo_input": torch.tensor([5.0, 6.0]),
                    "perturb_label": "B",
                },
            ]

        def __getitem__(self, index: int) -> dict:
            return self._rows[index]

    mean_sample = cohort_mean_sample(_FakeDataset(), [0, 1])
    assert torch.allclose(mean_sample["gene_input"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(mean_sample["cfo_input"], torch.tensor([4.0, 5.0]))
    assert mean_sample["cell_id"] == "cohort_mean_n2"


def test_perturbation_row_subset():
    class _FakeDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index: int) -> dict:
            return {"idx": index}

    base = _FakeDataset()
    subset = PerturbationRowSubset(base, [1, 3])
    assert len(subset) == 2
    assert subset[0]["idx"] == 1
    assert subset[1]["idx"] == 3


@pytest.mark.skipif(not H5AD.exists() or not GENE_UNIVERSE.exists(), reason="Norman h5ad missing")
def test_resolve_dataset_indices_on_h5ad():
    from regulus.io.h5ad_input import build_predict_dataset

    dataset = build_predict_dataset(
        H5AD,
        GENE_UNIVERSE,
        representation="delta",
        channels="cfo",
    )
    all_idx, n_all = resolve_dataset_indices(dataset, None)
    assert n_all == len(dataset)
    assert len(all_idx) == n_all

    first = dataset[0]
    cell_id = first["cell_id"]
    filt = f"cell_id == '{cell_id}'"
    one_idx, n_one = resolve_dataset_indices(dataset, filt)
    assert n_one == 1
    assert one_idx == [0] or dataset[one_idx[0]]["cell_id"] == cell_id
