"""Canonical relation names for the Regulus graph.

The tensor node key ``go`` is retained for checkpoint compatibility. Public
interfaces describe those selected GO biological-process nodes as CFO nodes.
"""

from __future__ import annotations

from typing import Final

EdgeType = tuple[str, str, str]

TF_GENE: Final[EdgeType] = ("tf", "regulates", "gene")
GENE_CFO: Final[EdgeType] = ("gene", "annotated_to", "go")
CELLTYPE_TF: Final[EdgeType] = ("celltype", "scenic_activity", "tf")
CELLTYPE_CFO: Final[EdgeType] = ("celltype", "llm_context", "go")
TF_CFO_LLM: Final[EdgeType] = ("tf", "llm_regulates", "go")

# Relation order is part of the HGT checkpoint contract. Do not reorder.
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

LEGACY_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    ("tf", "regulates", "gene"),
    ("gene", "in", "go"),
    ("celltype", "express", "tf"),
    ("celltype", "activates", "go"),
    ("tf", "regulates", "go"),
)

LEGACY_TO_CANONICAL: Final[dict[EdgeType, EdgeType]] = dict(
    zip(LEGACY_EDGE_TYPES, MESSAGE_PASSING_EDGE_TYPES)
)

PUBLIC_NODE_LABELS: Final[dict[str, str]] = {
    "tf": "TF",
    "gene": "Gene",
    "celltype": "CellType",
    "go": "CFO",
}
