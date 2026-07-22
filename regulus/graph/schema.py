"""Canonical node and relation names for the Regulus graph."""

from __future__ import annotations

from typing import Final

EdgeType = tuple[str, str, str]

TF_GENE: Final[EdgeType] = ("tf", "regulates", "gene")
GENE_CFO: Final[EdgeType] = ("gene", "annotated_to", "cfo")
CELLTYPE_TF: Final[EdgeType] = ("celltype", "scenic_activity", "tf")
CELLTYPE_CFO: Final[EdgeType] = ("celltype", "llm_context", "cfo")
TF_CFO_LLM: Final[EdgeType] = ("tf", "llm_regulates", "cfo")

# Keep relation order stable across training and inference.
MESSAGE_PASSING_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    TF_GENE,
    GENE_CFO,
    CELLTYPE_TF,
    CELLTYPE_CFO,
    TF_CFO_LLM,
)

# LLM TF-CFO confidence is evidence metadata, not a direct training target.
TRAINED_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    TF_GENE,
    GENE_CFO,
    CELLTYPE_TF,
    CELLTYPE_CFO,
)

PUBLIC_NODE_LABELS: Final[dict[str, str]] = {
    "tf": "TF",
    "gene": "Gene",
    "celltype": "CellType",
    "cfo": "CFO",
}
