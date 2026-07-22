"""Versioned graph-asset discovery, manifest generation, and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from regulus.graph.schema import MESSAGE_PASSING_EDGE_TYPES

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_ASSET_DIR = Path("assets/graph/regulus_graph_v1")
DEFAULT_MANIFEST_NAME = "manifest.json"

GRAPH_ASSET_FILES = {
    "metadata.json": "graph_metadata",
    "edges/edges_tf_gene.csv": "tf_gene_edges",
    "edges/edges_gene_cfo.csv": "gene_cfo_edges",
    "edges/edges_celltype_tf.csv": "celltype_tf_scenic_activity",
    "edges/edges_celltype_cfo.csv": "celltype_cfo_llm_context",
    "edges/edges_tf_cfo_llm.csv": "tf_cfo_llm_regulates",
    "embeddings/tf_family_onehot.npy": "tf_family_features",
    "embeddings/tf_celltype_expression.npy": "tf_celltype_activity_features",
    "embeddings/gene_geneformer_embeddings.npy": "gene_geneformer_features",
    "embeddings/celltype_gene_expression_pca.npy": "celltype_expression_features",
    "embeddings/cfo_text_embeddings.npy": "cfo_text_features",
    "embeddings/cfo_gene_counts.npy": "cfo_gene_count_features",
    "embeddings/gene_order.txt": "gene_order",
    "nodes/nodes_tf.csv": "tf_node_table",
    "nodes/nodes_gene.csv": "gene_node_table",
    "nodes/nodes_celltype.csv": "celltype_node_table",
    "nodes/nodes_cfo.csv": "cfo_node_table",
    "universes/gene_universe.csv": "gene_universe",
    "universes/tf_universe.csv": "tf_universe",
}


def resolve_graph_asset_dir(
    config: Optional[Mapping[str, Any]] = None,
    override: Optional[str | Path] = None,
) -> Path:
    """Resolve the graph asset root from a configuration or explicit override."""
    data_config = (config or {}).get("data", {})
    configured = data_config.get("graph_asset_dir")
    candidate = Path(override or configured or DEFAULT_GRAPH_ASSET_DIR).expanduser()
    return candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_description(path: Path) -> Dict[str, Any]:
    description: Dict[str, Any] = {"size_bytes": path.stat().st_size}
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        description.update(shape=list(values.shape), dtype=str(values.dtype))
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
            row_count = sum(1 for _ in reader)
        description.update(columns=columns, rows=row_count)
    return description


def build_graph_asset_manifest(
    graph_asset_dir: str | Path,
    *,
    source_commit: Optional[str] = None,
    source_tree_dirty: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create a manifest for canonical assets and retained provenance files."""
    root = Path(graph_asset_dir)
    missing = [relative for relative in GRAPH_ASSET_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing required graph assets: " + ", ".join(missing))

    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == DEFAULT_MANIFEST_NAME or relative.startswith("cache/"):
            continue
        entry = {
            "path": relative,
            "role": GRAPH_ASSET_FILES.get(relative, "provenance"),
            "required": relative in GRAPH_ASSET_FILES,
            "sha256": sha256_file(path),
            **_file_description(path),
        }
        files.append(entry)

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    combined = hashlib.sha256()
    for entry in files:
        combined.update(f"{entry['path']}:{entry['sha256']}\n".encode("utf-8"))
    manifest: Dict[str, Any] = {
        "asset_id": "regulus-graph-v1",
        "schema_version": "1.0",
        "asset_sha256": combined.hexdigest(),
        "public_node_labels": {
            "tf": "TF",
            "gene": "Gene",
            "celltype": "Cell type",
            "cfo": "CFO",
        },
        "node_counts": {
            "tf": metadata["n_tfs"],
            "gene": metadata["n_genes"],
            "celltype": metadata["n_celltypes"],
            "cfo": metadata["n_cfos"],
        },
        "relations": [list(edge_type) for edge_type in MESSAGE_PASSING_EDGE_TYPES],
        "precomputed_build_inputs": [
            "universes/gene_universe.csv",
            "universes/tf_universe.csv",
            "embeddings/gene_geneformer_embeddings.npy",
            "embeddings/gene_order.txt",
        ],
        "files": files,
    }
    if source_commit is not None:
        manifest["source_commit"] = source_commit
    if source_tree_dirty is not None:
        manifest["source_tree_dirty"] = source_tree_dirty
    return manifest


def write_graph_asset_manifest(
    graph_asset_dir: str | Path,
    *,
    source_commit: Optional[str] = None,
    source_tree_dirty: Optional[bool] = None,
) -> Path:
    root = Path(graph_asset_dir)
    manifest = build_graph_asset_manifest(
        root,
        source_commit=source_commit,
        source_tree_dirty=source_tree_dirty,
    )
    output = root / DEFAULT_MANIFEST_NAME
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def validate_graph_asset(
    graph_asset_dir: str | Path,
    *,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    validate_hashes: bool = True,
) -> Dict[str, Any]:
    """Validate required files and, when requested, their recorded hashes."""
    root = Path(graph_asset_dir)
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Graph asset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest.get("files", [])}
    missing_entries = [relative for relative in GRAPH_ASSET_FILES if relative not in entries]
    if missing_entries:
        raise ValueError("Manifest omits required graph assets: " + ", ".join(missing_entries))

    for relative, entry in entries.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Manifest path escapes graph asset root: {relative}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Graph asset size mismatch: {relative}")
        if validate_hashes and sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"Graph asset SHA-256 mismatch: {relative}")
    return manifest
