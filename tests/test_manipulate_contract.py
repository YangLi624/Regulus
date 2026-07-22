"""Virtual CFO manipulation output contracts."""

from types import SimpleNamespace

import pandas as pd
import torch

from regulus.manipulate.core import run_manipulation


class _Dataset:
    def __init__(self):
        self.rows = [
            {
                "cell_id": "cell_1",
                "gene_input": torch.tensor([1.0, 2.0]),
                "cfo_input": torch.tensor([0.2, 0.4]),
            },
            {
                "cell_id": "cell_2",
                "gene_input": torch.tensor([3.0, 4.0]),
                "cfo_input": torch.tensor([0.4, 0.6]),
            },
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Model:
    channels = "gene_cfo"
    candidate_perturbations = ["A", "B"]

    def __init__(self):
        self.dataset = _Dataset()
        self._hgt = SimpleNamespace(
            _loaded=True,
            cfo_nodes=pd.DataFrame({
                "cfo_index": [0, 1],
                "cfo_id": ["GO:1", "GO:2"],
                "cfo_name": ["one", "two"],
            }),
        )

    def _dataset(self, _path):
        return self.dataset

    def _logits(self, sample, _mode):
        cfo = sample["cfo_input"]
        if cfo.ndim == 1:
            cfo = cfo.unsqueeze(0)
        return cfo[:, :2]


def test_zero_cfo_edit_is_identity_and_does_not_mutate_anchor():
    model = _Model()
    original = model.dataset.rows[0]["cfo_input"].clone()
    frame = run_manipulation(
        model,
        cfo_targets=["GO:1"],
        cfo_delta=[0.0],
        anchor_h5ad="unused.h5ad",
        mode="joint",
        top_k=2,
    )
    assert (frame["score_delta"] == 0).all()
    assert (frame["rank_gain"] == 0).all()
    assert frame["runtime_mode"].eq("joint").all()
    assert frame["cfo_targets"].eq("GO:1").all()
    assert torch.equal(model.dataset.rows[0]["cfo_input"], original)
