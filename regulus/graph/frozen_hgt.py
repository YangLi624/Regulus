"""
冻结 HGT 图基座加载（自 celltype_inference / conditional_inference 剥离）。

仅负责：加载 checkpoint、重建 HeteroRegulatorNet、提供 node_embeddings 与图 metadata。
"""

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
    """
    冻结的 HGT 权重与节点嵌入，供扰动头训练 / 推理使用。
    """

    def __init__(
        self,
        model_path: str,
        processed_dir: str | Path = DEFAULT_GRAPH_ASSET_DIR,
        device: Optional[str] = None,
        embeddings_dir: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.graph_asset_dir = Path(processed_dir)
        self.processed_dir = self.graph_asset_dir
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[torch.nn.Module] = None
        self.data: Any = None
        self.config: Dict[str, Any] = {}
        self.go_nodes: Optional[pd.DataFrame] = None
        self.tf_nodes: Optional[pd.DataFrame] = None
        self.node_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False

    def load(self) -> "FrozenHGTBundle":
        """加载模型、异构图与节点嵌入（幂等）。"""
        if self._loaded:
            return self

        from regulus.graph.build import HeteroDataBuilder
        from regulus.graph.model import HeteroRegulatorNet, load_graph_state_dict

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint.get("config", {})

        builder = HeteroDataBuilder(str(self.processed_dir))
        self.data = builder.build_hetero_data(split_type=None, data_splitter=None)

        go_nodes_path = self.processed_dir / "nodes" / "nodes_go.csv"
        tf_nodes_path = self.processed_dir / "nodes" / "nodes_tf.csv"
        if go_nodes_path.exists():
            self.go_nodes = pd.read_csv(go_nodes_path)
        if tf_nodes_path.exists():
            self.tf_nodes = pd.read_csv(tf_nodes_path)

        model_config = self.config.get("model", {})
        feature_dims = builder.get_feature_dims()
        node_types = list(self.data.node_types)
        edge_types = list(self.data.edge_types)

        self.model = HeteroRegulatorNet(
            node_types=node_types,
            edge_types=edge_types,
            node_feature_dims=feature_dims,
            hidden_dim=model_config.get("hidden_dim", 128),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            use_reconstruction=model_config.get("use_reconstruction", True),
        ).to(self.device)
        load_graph_state_dict(self.model, checkpoint["model_state_dict"])
        self.model.eval()

        if "node_embeddings" in checkpoint:
            logger.info("Loading node embeddings from checkpoint...")
            self.node_embeddings = {
                k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in checkpoint["node_embeddings"].items()
            }
        elif self.embeddings_dir is not None and self.embeddings_dir.exists():
            logger.info("Loading node embeddings from %s", self.embeddings_dir)
            self.node_embeddings = {}
            for node_type in ("tf", "gene", "celltype", "go"):
                emb_path = self.embeddings_dir / f"{node_type}_embeddings.npy"
                if emb_path.exists():
                    emb = np.load(emb_path)
                    self.node_embeddings[node_type] = torch.as_tensor(
                        emb, dtype=torch.float32, device=self.device
                    )
        else:
            logger.info("Computing node embeddings from HGT encode...")
            self._compute_embeddings()

        self._loaded = True
        logger.info("FrozenHGTBundle loaded from %s", self.model_path)
        return self

    def _compute_embeddings(self) -> None:
        assert self.model is not None and self.data is not None
        self.model.eval()
        with torch.no_grad():
            self.data = self.data.to(self.device)
            x_dict = {
                nt: self.data[nt].x
                for nt in self.model.node_types
                if nt in self.data.node_types
            }
            edge_index_dict = {
                et: self.data[et].edge_index
                for et in self.model.edge_types
                if et in self.data.edge_types
            }
            self.node_embeddings = self.model.encode(
                x_dict, edge_index_dict, edge_mask_dict=None
            )
