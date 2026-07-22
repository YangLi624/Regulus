"""Load a frozen Regulus graph encoder and its node embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from regulus.graph.assets import DEFAULT_GRAPH_ASSET_DIR

logger = logging.getLogger(__name__)


class FrozenHGTBundle:
    """Frozen graph weights, node metadata, and embeddings for perturbation models."""

    def __init__(
        self,
        model_path: str,
        graph_asset_dir: str | Path = DEFAULT_GRAPH_ASSET_DIR,
        device: Optional[str] = None,
        embeddings_dir: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.graph_asset_dir = Path(graph_asset_dir)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model: Optional[torch.nn.Module] = None
        self.data: Any = None
        self.config: Dict[str, Any] = {}
        self.cfo_nodes: Optional[pd.DataFrame] = None
        self.tf_nodes: Optional[pd.DataFrame] = None
        self.node_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False

    def load(self) -> "FrozenHGTBundle":
        """Load the graph, model weights, node metadata, and embeddings once."""
        if self._loaded:
            return self

        from regulus.graph.build import HeteroDataBuilder
        from regulus.graph.model import HeteroRegulatorNet

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint.get("config", {})

        builder = HeteroDataBuilder(self.graph_asset_dir)
        self.data = builder.build_hetero_data()

        cfo_path = self.graph_asset_dir / "nodes" / "nodes_cfo.csv"
        tf_path = self.graph_asset_dir / "nodes" / "nodes_tf.csv"
        if cfo_path.exists():
            self.cfo_nodes = pd.read_csv(cfo_path)
        if tf_path.exists():
            self.tf_nodes = pd.read_csv(tf_path)

        model_config = self.config.get("model", {})
        self.model = HeteroRegulatorNet(
            node_types=list(self.data.node_types),
            edge_types=list(self.data.edge_types),
            node_feature_dims=builder.get_feature_dims(),
            hidden_dim=model_config.get("hidden_dim", 128),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            use_reconstruction=model_config.get("use_reconstruction", True),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        if "node_embeddings" in checkpoint:
            self.node_embeddings = {
                key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for key, value in checkpoint["node_embeddings"].items()
            }
        elif self.embeddings_dir is not None and self.embeddings_dir.exists():
            self.node_embeddings = {}
            for node_type in ("tf", "gene", "celltype", "cfo"):
                path = self.embeddings_dir / f"{node_type}_embeddings.npy"
                if path.exists():
                    self.node_embeddings[node_type] = torch.as_tensor(
                        np.load(path), dtype=torch.float32, device=self.device
                    )
        else:
            self._compute_embeddings()

        self._loaded = True
        logger.info("Loaded frozen graph model from %s", self.model_path)
        return self

    def _compute_embeddings(self) -> None:
        """Encode every graph node when a checkpoint has no cached embeddings."""
        assert self.model is not None and self.data is not None
        self.model.eval()
        with torch.no_grad():
            self.data = self.data.to(self.device)
            x_dict = {
                node_type: self.data[node_type].x
                for node_type in self.model.node_types
                if node_type in self.data.node_types
            }
            edge_index_dict = {
                edge_type: self.data[edge_type].edge_index
                for edge_type in self.model.edge_types
                if edge_type in self.data.edge_types
            }
            self.node_embeddings = self.model.encode(x_dict, edge_index_dict)
