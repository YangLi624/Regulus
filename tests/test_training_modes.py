import random

import pytest
import torch

from regulus.perturb.training_modes import (
    ENCODE_MODES,
    compute_topk_accuracy,
    normalize_mode_schedule,
    sample_encode_mode,
)


def test_normalize_schedule():
    result = normalize_mode_schedule({"joint": 2, "gene_only": 1, "cfo_only": 1})
    assert sum(result.values()) == pytest.approx(1.0)
    assert set(result) == set(ENCODE_MODES)


def test_unknown_mode_fails():
    with pytest.raises(ValueError):
        normalize_mode_schedule({"go_only": 1})


def test_sample_deterministic_single_mode():
    assert sample_encode_mode({"gene_only": 1}, random.Random(1)) == "gene_only"


def test_topk_handles_fewer_than_ten_candidates():
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    metrics = compute_topk_accuracy(logits, torch.tensor([1]))
    assert metrics["top1_acc"] == 1.0
    assert metrics["top10_acc"] == 1.0
