"""Heterogeneous graph encoder used to pretrain Regulus node prototypes."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

from regulus.graph.schema import CELLTYPE_CFO, CELLTYPE_TF, GENE_CFO, TF_GENE

logger = logging.getLogger(__name__)

class HeteroRegulatorNet(nn.Module):
    """HGT encoder with four relation-reconstruction objectives."""

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_feature_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_reconstruction: bool = True,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.use_reconstruction = use_reconstruction
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.node_projections = nn.ModuleDict(
            {
                node_type: Linear(node_feature_dims[node_type], hidden_dim)
                for node_type in node_types
            }
        )
        self.hgt_layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(node_types, edge_types),
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in node_types}
        )
        self.dropout = nn.Dropout(dropout)

        if use_reconstruction:
            self.recon_decoders = nn.ModuleDict(
                {
                    "tf_gene": BilinearDecoder(hidden_dim),
                    "gene_cfo": BilinearDecoder(hidden_dim),
                    "celltype_tf": BilinearDecoder(hidden_dim),
                    "celltype_cfo": BilinearDecoder(hidden_dim),
                }
            )
        else:
            self.recon_decoders = nn.ModuleDict()

        logger.info(
            "Initialized HeteroRegulatorNet with %d HGT layers, hidden_dim=%d, heads=%d",
            num_layers,
            hidden_dim,
            num_heads,
        )

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_mask_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode node features with relation-aware message passing."""
        h_dict = {}
        for node_type, features in x_dict.items():
            projected_input = features.detach() if node_type == "celltype" else features
            h_dict[node_type] = self.dropout(
                F.relu(self.node_projections[node_type](projected_input))
            )

        masked_edges = edge_index_dict
        if edge_mask_dict is not None:
            masked_edges = {}
            for edge_type, edge_index in edge_index_dict.items():
                mask = edge_mask_dict.get(edge_type)
                if mask is not None and mask.dim() == 1 and mask.shape[0] == edge_index.shape[1]:
                    masked_edges[edge_type] = edge_index[:, mask]
                else:
                    masked_edges[edge_type] = edge_index

        def hgt_layer_forward(
            current: Dict[str, torch.Tensor],
            layer: HGTConv,
            edges: Dict[Tuple[str, str, str], torch.Tensor],
            norms: nn.ModuleDict,
            dropout_layer: nn.Dropout,
        ) -> Dict[str, torch.Tensor]:
            updated = layer(current, edges)
            return {
                node_type: norms[node_type](
                    node_features + dropout_layer(updated[node_type])
                )
                if node_type in updated
                else node_features
                for node_type, node_features in current.items()
            }

        for layer in self.hgt_layers:
            if self.training and self.use_gradient_checkpointing:
                h_dict = checkpoint(
                    hgt_layer_forward,
                    h_dict,
                    layer,
                    masked_edges,
                    self.layer_norms,
                    self.dropout,
                    use_reentrant=False,
                )
            else:
                h_dict = hgt_layer_forward(
                    h_dict,
                    layer,
                    masked_edges,
                    self.layer_norms,
                    self.dropout,
                )
        return h_dict

    @staticmethod
    def generate_edge_mask(
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        mask_ratio: float = 0.1,
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Generate graph views by masking every message-passing relation."""
        masks = {}
        for edge_type, edge_index in edge_index_dict.items():
            num_edges = edge_index.shape[1]
            num_mask = int(num_edges * mask_ratio)
            mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            if 0 < num_mask < num_edges:
                mask[torch.randperm(num_edges, device=edge_index.device)[:num_mask]] = False
            masks[edge_type] = mask
        return masks

    @staticmethod
    def _reconstruction_edges(data: HeteroData, edge_type: Tuple[str, str, str]) -> torch.Tensor:
        store = data[edge_type]
        return store.recon_edge_index if hasattr(store, "recon_edge_index") else store.edge_index

    def decode_reconstruction(
        self,
        h_dict: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """Decode only the four relations used as direct training targets."""
        if not self.use_reconstruction:
            return {}

        relation_tasks = (
            (TF_GENE, "tf_gene_recon", "tf_gene", "tf", "gene"),
            (GENE_CFO, "gene_cfo_recon", "gene_cfo", "gene", "cfo"),
            (CELLTYPE_TF, "celltype_tf_recon", "celltype_tf", "celltype", "tf"),
            (CELLTYPE_CFO, "celltype_cfo_recon", "celltype_cfo", "celltype", "cfo"),
        )
        return {
            output_key: self.recon_decoders[decoder_key](
                h_dict[src_type],
                h_dict[dst_type],
                self._reconstruction_edges(data, edge_type),
            )
            for edge_type, output_key, decoder_key, src_type, dst_type in relation_tasks
            if edge_type in data.edge_types
        }

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Encode the graph and return reconstruction logits plus node embeddings."""
        x_dict = {
            node_type: data[node_type].x
            for node_type in self.node_types
            if node_type in data.node_types
        }
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type not in data.edge_types:
                continue
            store = data[edge_type]
            edge_index_dict[edge_type] = (
                store.edge_index_encoder
                if hasattr(store, "edge_index_encoder")
                else store.edge_index
            )

        h_dict = self.encode(x_dict, edge_index_dict)
        results = self.decode_reconstruction(h_dict, data)
        results["node_embeddings"] = h_dict
        return results


class BilinearDecoder(nn.Module):
    """Relation-specific bilinear edge decoder."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src = src_embeddings[edge_index[0]]
        dst = dst_embeddings[edge_index[1]]
        return self.bilinear(src, dst).squeeze(-1)
