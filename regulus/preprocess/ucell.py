"""Compute CFO activity matrices with pyUCell."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)

CFO_ACTIVITY_OBSM_KEY = "X_cfo_activity"
REGULUS_CFO_IDS_UNS_KEY = "regulus_cfo_ids"
def missing_cfo_activity_message() -> str:
    return (
        f"CFO activity matrix not found in obsm[{CFO_ACTIVITY_OBSM_KEY!r}]. Run "
        "`regulus preprocess -i <input.h5ad> --bundle-path <bundle>` first."
    )


def load_cfo_gene_sets(csv_path: str | Path) -> dict[str, list[str]]:
    """Load ordered CFO gene sets from columns ``cfo_id`` and ``genes``."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    if "cfo_id" not in frame.columns or "genes" not in frame.columns:
        raise ValueError(
            f"Expected columns 'cfo_id' and 'genes' in {path}, got {frame.columns.tolist()}"
        )
    gene_sets: dict[str, list[str]] = OrderedDict()
    for _, row in frame.iterrows():
        cfo_id = str(row["cfo_id"])
        genes = [value.strip() for value in str(row["genes"]).split(";") if value.strip()]
        if genes:
            gene_sets[cfo_id] = genes
    return gene_sets


def load_cfo_gene_sets_from_graph_asset(graph_asset_dir: str | Path) -> dict[str, list[str]]:
    """Reconstruct ordered CFO gene sets from a released graph asset."""
    root = Path(graph_asset_dir)
    genes = pd.read_csv(root / "nodes" / "nodes_gene.csv")
    cfos = pd.read_csv(root / "nodes" / "nodes_cfo.csv")
    edges = pd.read_csv(root / "edges" / "edges_gene_cfo.csv")
    required = {
        "nodes_gene.csv": (genes, {"gene_index", "gene_symbol"}),
        "nodes_cfo.csv": (cfos, {"cfo_index", "cfo_id"}),
        "edges_gene_cfo.csv": (edges, {"src_node_id", "dst_node_id"}),
    }
    for name, (frame, columns) in required.items():
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    gene_names = dict(zip(genes["gene_index"].astype(int), genes["gene_symbol"].astype(str)))
    members: dict[int, list[str]] = {}
    for gene_index, cfo_index in zip(edges["src_node_id"], edges["dst_node_id"]):
        symbol = gene_names.get(int(gene_index))
        if symbol is not None:
            members.setdefault(int(cfo_index), []).append(symbol)

    gene_sets: dict[str, list[str]] = OrderedDict()
    for _, row in cfos.sort_values("cfo_index").iterrows():
        cfo_index = int(row["cfo_index"])
        values = members.get(cfo_index, [])
        if values:
            gene_sets[str(row["cfo_id"])] = values
    return gene_sets


def get_cfo_activity_n_cfo(hca) -> Optional[int]:
    matrix = getattr(hca, "obsm", {}).get(CFO_ACTIVITY_OBSM_KEY)
    return None if matrix is None else int(matrix.shape[1])


def require_cfo_activity_matrix(hca, *, context: str = "") -> np.ndarray:
    if CFO_ACTIVITY_OBSM_KEY not in getattr(hca, "obsm", {}):
        prefix = f"{context}: " if context else ""
        raise KeyError(prefix + missing_cfo_activity_message())
    return np.asarray(hca.obsm[CFO_ACTIVITY_OBSM_KEY])


def get_cfo_activity_row(hca, row: int) -> np.ndarray:
    return np.asarray(require_cfo_activity_matrix(hca)[row]).flatten().astype(np.float32)


def validate_cfo_activity(
    hca,
    expected_cfo_ids: int | list[str] | tuple[str, ...],
    *,
    context: str = "",
) -> None:
    matrix = require_cfo_activity_matrix(hca, context=context)
    expected_count = (
        int(expected_cfo_ids)
        if isinstance(expected_cfo_ids, int)
        else len(expected_cfo_ids)
    )
    if matrix.shape[1] != expected_count:
        raise ValueError(
            f"{context}: CFO activity has {matrix.shape[1]} columns; expected {expected_count}. "
            "Re-run `regulus preprocess` with the matching bundle."
        )
    if not isinstance(expected_cfo_ids, int):
        observed = hca.uns.get(REGULUS_CFO_IDS_UNS_KEY)
        if observed is None:
            raise ValueError(f"{context}: CFO identifiers are missing from adata.uns")
        if list(map(str, observed)) != list(map(str, expected_cfo_ids)):
            raise ValueError(f"{context}: CFO identifier order does not match the graph asset")


def compute_ucell_cfo_activity(
    adata: sc.AnnData,
    gene_sets: dict[str, list[str]],
) -> list[str]:
    """Compute CFO activity and remove temporary pyUCell columns from ``obs``."""
    try:
        import pyucell as uc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyucell is required for CFO activity preprocessing") from exc

    signatures = {cfo_id.replace(":", "_"): genes for cfo_id, genes in gene_sets.items()}
    uc.compute_ucell_scores(adata, signatures=signatures, suffix="_UCell", n_jobs=1)

    cfo_ids = list(gene_sets)
    score_columns: list[str] = []
    scores: list[np.ndarray] = []
    for cfo_id in cfo_ids:
        safe_name = cfo_id.replace(":", "_")
        exact = f"{safe_name}_UCell"
        candidates = [
            str(column)
            for column in adata.obs.columns
            if safe_name in str(column) and "ucell" in str(column).lower()
        ]
        column = (
            exact
            if exact in adata.obs.columns
            else (candidates[0] if len(candidates) == 1 else None)
        )
        if column is None:
            raise KeyError(f"pyUCell did not create a unique score column for CFO {cfo_id}")
        score_columns.append(column)
        scores.append(adata.obs[column].to_numpy(dtype=np.float32))

    adata.obsm[CFO_ACTIVITY_OBSM_KEY] = np.stack(scores, axis=0).T.astype(np.float32)
    adata.uns[REGULUS_CFO_IDS_UNS_KEY] = cfo_ids
    adata.uns["regulus_cfo_activity_method"] = "pyucell"
    adata.obs.drop(columns=list(dict.fromkeys(score_columns)), inplace=True)
    return cfo_ids


def preprocess_h5ad_ucell(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cfo_gene_sets: str | Path | None = None,
    graph_asset_dir: str | Path | None = None,
    layer: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Compute CFO activities and write a new H5AD file."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"H5AD input not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass overwrite=True to replace it"
        )

    source_csv = cfo_gene_sets
    if (source_csv is None) == (graph_asset_dir is None):
        raise ValueError("Provide exactly one of cfo_gene_sets or graph_asset_dir")
    gene_sets = (
        load_cfo_gene_sets(source_csv)
        if source_csv is not None
        else load_cfo_gene_sets_from_graph_asset(graph_asset_dir)  # type: ignore[arg-type]
    )

    adata = sc.read_h5ad(str(input_path))
    original_x = adata.X
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers")
        adata.X = adata.layers[layer]
    try:
        compute_ucell_cfo_activity(adata, gene_sets)
    finally:
        adata.X = original_x

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(output_path))
    logger.info("Saved CFO activity matrix to %s", output_path)
    return output_path
