"""Tests for frozen candidate prototypes and matching logits."""

from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
GENE_ORDER = ROOT / "assets/graph/regulus_graph_v1/embeddings/gene_order.txt"


class _FakeHGT:
    def __init__(self, n_tf: int = 4, n_gene: int = 6, dim: int = 8):
        self.tf_nodes = pd.DataFrame({
            "tf_symbol": [f"TF{i}" for i in range(n_tf)],
            "tf_index": list(range(n_tf)),
        })
        self.node_embeddings = {
            "tf": torch.arange(n_tf * dim, dtype=torch.float32).reshape(n_tf, dim),
            "gene": torch.arange(n_gene * dim, dtype=torch.float32).reshape(n_gene, dim) + 100,
        }


def test_load_gene_symbol_to_idx_from_gene_order():
    from regulus.perturb.prototype_utils import load_gene_symbol_to_idx

    if not GENE_ORDER.exists():
        pytest.skip("gene_order.txt not in repo")
    mapping = load_gene_symbol_to_idx(
        ROOT / "missing_gene_universe.csv",
        graph_asset_dir=ROOT / "assets/graph/regulus_graph_v1",
    )
    assert mapping["A1BG"] == 0
    assert mapping["A1CF"] == 1


def test_build_candidate_prototypes_mixed(tmp_path):
    from regulus.perturb.prototype_utils import build_candidate_prototypes

    gene_universe = tmp_path / "gene_universe.csv"
    pd.DataFrame({
        "gene_symbol": [f"G{i}" for i in range(6)],
        "gene_index": list(range(6)),
    }).to_csv(gene_universe, index=False)
    pd.DataFrame({
        "tf_symbol": [f"TF{i}" for i in range(4)],
        "tf_index": list(range(4)),
    }).to_csv(tmp_path / "tf_universe.csv", index=False)

    hgt = _FakeHGT()
    candidates = ["TF0", "TF1", "G2", "G3"]
    is_tf = {"TF0": True, "TF1": True, "G2": False, "G3": False}
    prototypes, kinds = build_candidate_prototypes(
        candidates, is_tf, hgt, gene_universe, device=torch.device("cpu")
    )
    assert prototypes.shape == (4, 8)
    assert kinds == ["tf", "tf", "gene", "gene"]
    assert torch.allclose(prototypes[0], hgt.node_embeddings["tf"][0])
    assert torch.allclose(prototypes[2], hgt.node_embeddings["gene"][2])


def test_build_candidate_prototypes_missing_gene_raises(tmp_path):
    from regulus.perturb.prototype_utils import build_candidate_prototypes

    gene_universe = tmp_path / "gene_universe.csv"
    pd.DataFrame({"gene_symbol": ["G0"], "gene_index": [0]}).to_csv(gene_universe, index=False)
    pd.DataFrame({"tf_symbol": ["TF0"], "tf_index": [0]}).to_csv(
        tmp_path / "tf_universe.csv", index=False
    )
    with pytest.raises(KeyError):
        build_candidate_prototypes(
            ["UNKNOWN_GENE"], {"UNKNOWN_GENE": False}, _FakeHGT(1, 1), gene_universe
        )


def test_prototype_matching_head_forward():
    from regulus.perturb.models import PrototypeMatchingHead

    head = PrototypeMatchingHead(16, torch.randn(5, 16), use_bilinear=False)
    assert head(torch.randn(3, 16)).shape == (3, 5)
