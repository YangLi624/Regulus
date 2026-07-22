"""Unified architecture routing and release-model recipes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from regulus.io.h5ad_input import build_predict_dataset
from regulus.perturb.data import PerturbationDataset
from regulus.perturb.model import RegulusPerturbationModel
from regulus.perturb.models import GeneTokenTransformerEncoder, JointCrossTransformerEncoder
from regulus.perturb.spec import PerturbModelSpec, normalize_mode

ROOT = Path(__file__).parent.parent


def _embeddings():
    torch.manual_seed(1)
    return torch.randn(7, 8), torch.randn(5, 8), torch.randn(3, 8)


def test_gene_channels_route_to_gene_token_encoder():
    gene, _, prototypes = _embeddings()
    model = RegulusPerturbationModel(
        channels="gene", head="mlp", h_gene=gene, h_cfo=None,
        d_cond=8, n_candidates=3, gene_topk=4, n_layers=1,
        n_heads=2, gene_n_heads=2, dropout=0,
    )
    assert isinstance(model.gene_encoder, GeneTokenTransformerEncoder)
    assert model.joint_encoder is None
    assert not any("delta" in key for key in model.gene_encoder.state_dict())
    assert model(gene_input=torch.randn(2, 7), mode="gene_only").shape == (2, 3)


def test_cfo_and_joint_channels_route_to_joint_encoder():
    gene, cfo, prototypes = _embeddings()
    cfo_model = RegulusPerturbationModel(
        channels="cfo", head="prototype_matching", h_gene=gene, h_cfo=cfo,
        h_prototypes=prototypes, d_cond=8, n_candidates=3,
        gene_topk=4, cfo_topk=3, n_layers=1, n_heads=2, dropout=0,
    )
    assert isinstance(cfo_model.joint_encoder, JointCrossTransformerEncoder)
    assert cfo_model(cfo_input=torch.randn(2, 5), mode="cfo_only").shape == (2, 3)

    joint_model = RegulusPerturbationModel(
        channels="gene_cfo", head="prototype_matching", h_gene=gene, h_cfo=cfo,
        h_prototypes=prototypes, d_cond=8, n_candidates=3,
        gene_topk=4, cfo_topk=3, n_layers=1, n_heads=2, dropout=0,
    )
    encoder_id = id(joint_model.joint_encoder)
    for mode in ("joint", "gene_only", "cfo_only"):
        logits = joint_model(
            gene_input=torch.randn(2, 7), cfo_input=torch.randn(2, 5), mode=mode
        )
        assert logits.shape == (2, 3)
        assert id(joint_model.joint_encoder) == encoder_id


def test_channel_mode_is_constrained_by_trained_channels():
    assert normalize_mode("gene_only", "gene_cfo") == "gene_only"
    with pytest.raises(ValueError):
        normalize_mode("cfo_only", "gene")


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("perturb_norman_post_state.yaml", ("post_state", "gene_cfo", "prototype_matching")),
        ("perturb_schmidt_post_state.yaml", ("post_state", "gene_cfo", "prototype_matching")),
        ("perturb_tian_crispra_post_state.yaml", ("post_state", "gene_cfo", "prototype_matching")),
        ("perturb_joung_cfo_post_state.yaml", ("post_state", "cfo", "prototype_matching")),
    ],
)
def test_released_figure_recipes(filename, expected):
    text = (ROOT / "configs" / filename).read_text(encoding="utf-8")
    assert "tasks/github_release" not in text
    config = yaml.safe_load(text)
    spec = PerturbModelSpec.from_config(config)
    assert (spec.representation, spec.channels, spec.head) == expected
    assert config["training"]["loss"] == "cross_entropy"


def test_dataset_does_not_require_labels_or_transform_values(tmp_path):
    ad = pytest.importorskip("anndata")
    matrix = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(matrix)
    adata.var_names = ["A", "B"]
    adata.obs_names = ["cell_1", "cell_2"]
    path = tmp_path / "prediction.h5ad"
    adata.write_h5ad(path)
    universe = tmp_path / "gene_universe.csv"
    pd.DataFrame({"gene_symbol": ["A", "B"], "gene_index": [0, 1]}).to_csv(
        universe, index=False
    )
    dataset = PerturbationDataset(
        path, universe, representation="delta", channels="gene", require_labels=False
    )
    np.testing.assert_array_equal(dataset[0]["gene_input"].numpy(), matrix[0])
    assert "perturb_label" not in dataset[0]


def test_prediction_keeps_control_rows_while_training_excludes_them(tmp_path):
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    adata.var_names = ["A", "B"]
    adata.obs_names = ["control_cell", "treated_cell"]
    adata.obs["perturbation"] = ["control", "GENE_A"]
    path = tmp_path / "control_rows.h5ad"
    adata.write_h5ad(path)
    universe = tmp_path / "gene_universe.csv"
    pd.DataFrame({"gene_symbol": ["A", "B"], "gene_index": [0, 1]}).to_csv(
        universe, index=False
    )

    prediction = build_predict_dataset(
        path, universe, representation="post_state", channels="gene"
    )
    training = PerturbationDataset(
        path,
        universe,
        representation="post_state",
        channels="gene",
        require_labels=True,
    )

    assert [prediction[index]["cell_id"] for index in range(len(prediction))] == [
        "control_cell",
        "treated_cell",
    ]
    assert [training[index]["cell_id"] for index in range(len(training))] == [
        "treated_cell"
    ]
