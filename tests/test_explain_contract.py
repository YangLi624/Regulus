"""Attribution and evidence-source contracts."""

import torch

from regulus.explain.attribution import attribute_batch
from regulus.explain.evidence import (
    LLM_CONTEXT,
    REFERENCE_SUPPORTED,
    EvidenceGraphBuilder,
)
from regulus.perturb.model import RegulusPerturbationModel


def _joint_model():
    torch.manual_seed(2)
    return RegulusPerturbationModel(
        channels="gene_cfo",
        head="mlp",
        h_gene=torch.randn(6, 8),
        h_cfo=torch.randn(4, 8),
        d_cond=8,
        n_candidates=3,
        gene_topk=4,
        cfo_topk=3,
        n_layers=1,
        n_heads=2,
        dropout=0,
    ).eval()


def test_both_channel_gradient_x_input():
    result = attribute_batch(
        _joint_model(),
        gene_input=torch.randn(2, 6),
        cfo_input=torch.randn(2, 4),
        mode="joint",
        attribution_channel="both",
        attribution_method="gradient_x_input",
    )
    assert set(result.signed_attributions) == {"gene", "cfo"}
    assert result.signed_attributions["gene"].shape == (2, 6)
    assert result.signed_attributions["cfo"].shape == (2, 4)


def test_integrated_gradients_keeps_explicit_target_and_zero_input_is_zero():
    target = torch.tensor([1])
    result = attribute_batch(
        _joint_model(),
        gene_input=torch.zeros(1, 6),
        cfo_input=torch.randn(1, 4),
        mode="joint",
        attribution_channel="gene",
        attribution_method="integrated_gradients",
        target_indices=target,
        ig_steps=4,
    )
    assert result.target_indices.tolist() == [1]
    assert torch.count_nonzero(result.signed_attributions["gene"]) == 0


def test_channel_must_be_active_in_runtime_mode():
    try:
        attribute_batch(
            _joint_model(),
            gene_input=torch.randn(1, 6),
            cfo_input=torch.randn(1, 4),
            mode="gene_only",
            attribution_channel="cfo",
        )
    except ValueError as error:
        assert "runtime mode" in str(error)
    else:
        raise AssertionError("inactive attribution channel was accepted")


def test_evidence_uses_two_descriptive_source_classes():
    builder = object.__new__(EvidenceGraphBuilder)
    builder.gene_symbols = ["G1"]
    builder.gene_symbol_to_idx = {"G1": 0}
    builder.tf_idx_to_symbol = {0: "TF1"}
    builder.tf_symbol_to_idx = {"TF1": 0}
    builder.cfo_idx_to_id = {0: "GO:1"}
    builder.cfo_idx_to_name = {0: "CFO one"}
    builder.cfo_label_mode = "id"
    builder.gene_to_tf = {0: {0: 1.0}}
    builder.gene_to_cfo = {0: {0: 1.0}}
    builder.tf_to_cfo_llm = {0: {0: 0.8}}
    paths = builder.trace_from_genes([0], [0.5], candidate_names=["TF1"])
    sources = {path.evidence_source for path in paths}
    assert sources == {REFERENCE_SUPPORTED, LLM_CONTEXT}
    llm_paths = [path for path in paths if path.evidence_source == LLM_CONTEXT]
    assert all(path.gene == "G1" for path in llm_paths)
    assert all(path.candidate == "TF1" for path in paths)
    assert not any(source in {"hard", "weak"} for source in sources)
