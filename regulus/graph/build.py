"""Build the Regulus PyTorch Geometric graph from processed assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from regulus.graph.assets import (
    DEFAULT_GRAPH_ASSET_DIR,
    DEFAULT_MANIFEST_NAME,
    validate_graph_asset,
)
from regulus.graph.schema import CELLTYPE_CFO, CELLTYPE_TF, GENE_CFO, TF_CFO_LLM, TF_GENE


class HeteroDataBuilder:
    """Load the fixed node universes, features, and five graph relations."""

    def __init__(
        self,
        graph_asset_dir: str | Path = DEFAULT_GRAPH_ASSET_DIR,
        *,
        manifest_name: Optional[str] = None,
        validate_hashes: bool = False,
    ) -> None:
        self.graph_asset_dir = Path(graph_asset_dir)
        self.manifest = None
        if manifest_name:
            self.manifest = validate_graph_asset(
                self.graph_asset_dir,
                manifest_name=manifest_name or DEFAULT_MANIFEST_NAME,
                validate_hashes=validate_hashes,
            )
        with open(self.graph_asset_dir / "metadata.json", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.node_counts = {
            "tf": self.metadata["n_tfs"],
            "gene": self.metadata["n_genes"],
            "celltype": self.metadata["n_celltypes"],
            "cfo": self.metadata["n_cfos"],
        }

    def _load_array(self, relative_path: str, expected_rows: int) -> np.ndarray:
        values = np.load(self.graph_asset_dir / relative_path)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] != expected_rows and values.shape[1] == expected_rows:
            values = values.T
        if values.shape[0] != expected_rows:
            raise ValueError(
                f"{relative_path} has shape {values.shape}; expected {expected_rows} rows"
            )
        return values

    def _load_node_features(self) -> Dict[str, torch.Tensor]:
        tf_features = np.concatenate(
            [
                self._load_array("embeddings/tf_family_onehot.npy", self.node_counts["tf"]),
                self._load_array(
                    "embeddings/tf_celltype_expression.npy", self.node_counts["tf"]
                ),
            ],
            axis=1,
        )
        return {
            "tf": torch.as_tensor(tf_features, dtype=torch.float32),
            "gene": torch.as_tensor(
                self._load_array(
                    "embeddings/gene_geneformer_embeddings.npy", self.node_counts["gene"]
                ),
                dtype=torch.float32,
            ),
            "celltype": torch.as_tensor(
                self._load_array(
                    "embeddings/celltype_gene_expression_pca.npy",
                    self.node_counts["celltype"],
                ),
                dtype=torch.float32,
            ),
            "cfo": torch.as_tensor(
                np.concatenate(
                    [
                        self._load_array(
                            "embeddings/cfo_text_embeddings.npy", self.node_counts["cfo"]
                        ),
                        self._load_array(
                            "embeddings/cfo_gene_counts.npy", self.node_counts["cfo"]
                        ),
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
            ),
        }

    def _read_edges(self, filename: str) -> Dict[str, torch.Tensor]:
        frame = pd.read_csv(self.graph_asset_dir / "edges" / filename)
        edge_index = torch.as_tensor(
            frame[["src_node_id", "dst_node_id"]].to_numpy().T,
            dtype=torch.long,
        )
        return {
            "edge_index": edge_index,
            "edge_attr": torch.as_tensor(frame["weight"].to_numpy(), dtype=torch.float32),
        }

    def _load_edges(self) -> Dict:
        edge_files = (
            (TF_GENE, "edges_tf_gene.csv", True),
            (GENE_CFO, "edges_gene_cfo.csv", True),
            (CELLTYPE_TF, "edges_celltype_tf.csv", True),
            (CELLTYPE_CFO, "edges_celltype_cfo.csv", True),
            (TF_CFO_LLM, "edges_tf_cfo_llm.csv", False),
        )
        edges = {}
        for edge_type, filename, required in edge_files:
            path = self.graph_asset_dir / "edges" / filename
            if path.exists():
                edges[edge_type] = self._read_edges(filename)
            elif required:
                raise FileNotFoundError(path)
        return edges

    def build_hetero_data(self) -> HeteroData:
        """Build the complete heterogeneous graph."""
        data = HeteroData()
        for node_type, features in self._load_node_features().items():
            data[node_type].x = features
            data[node_type].num_nodes = self.node_counts[node_type]
        data["gene"].node_id = torch.arange(self.node_counts["gene"])
        for edge_type, values in self._load_edges().items():
            data[edge_type].edge_index = values["edge_index"]
            data[edge_type].edge_attr = values["edge_attr"]
        return data

    def get_feature_dims(self) -> Dict[str, int]:
        return {
            node_type: int(features.shape[1])
            for node_type, features in self._load_node_features().items()
        }

    def get_data_statistics(self) -> Dict:
        edges = self._load_edges()
        edge_counts = {
            edge_type: values["edge_index"].shape[1]
            for edge_type, values in edges.items()
        }
        edge_counts["total"] = sum(edge_counts.values())
        node_counts = dict(self.node_counts)
        node_counts["total"] = sum(node_counts.values())
        return {
            "num_nodes": node_counts,
            "num_edges": edge_counts,
            "feature_dims": self.get_feature_dims(),
            "metadata": self.metadata,
            "graph_asset": {
                "asset_dir": str(self.graph_asset_dir),
                "asset_id": (self.manifest or {}).get("asset_id"),
                "asset_sha256": (self.manifest or {}).get("asset_sha256"),
            },
        }

    @staticmethod
    def save_hetero_data(data: HeteroData, save_path: str) -> None:
        """Optionally materialize the complete graph as one reviewable asset."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)

    @staticmethod
    def load_hetero_data(load_path: str) -> HeteroData:
        return torch.load(load_path, weights_only=False)
