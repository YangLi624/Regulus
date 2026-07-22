"""Losses used by Regulus heterogeneous graph pretraining."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from regulus.graph.schema import CELLTYPE_CFO, CELLTYPE_TF, GENE_CFO, TF_GENE

logger = logging.getLogger(__name__)


class HeteroLoss(nn.Module):
    """Combine four reconstruction objectives with graph contrastive learning."""

    def __init__(
        self,
        tf_gene_recon_weight: float = 0.2,
        gene_cfo_recon_weight: float = 0.2,
        celltype_tf_recon_weight: float = 0.2,
        celltype_cfo_recon_weight: float = 0.2,
        contrast_weight: float = 0.25,
        contrast_temperature: float = 0.07,
        contrast_num_negatives: int = 4096,
        contrast_batch_size: int = 2048,
        contrast_skip_gene: bool = False,
        cross_type_align_weight: float = 0.5,
        align_celltype_cfo_weight: float = 0.25,
        align_tf_gene_weight: float = 0.15,
        align_gene_cfo_weight: float = 0.20,
        align_celltype_tf_weight: float = 0.15,
        recon_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight

        self.tf_gene_recon_weight = tf_gene_recon_weight
        self.gene_cfo_recon_weight = gene_cfo_recon_weight
        self.celltype_tf_recon_weight = celltype_tf_recon_weight
        self.celltype_cfo_recon_weight = celltype_cfo_recon_weight

        self.cross_type_align_weight = cross_type_align_weight
        self.align_celltype_cfo_weight = align_celltype_cfo_weight
        self.align_tf_gene_weight = align_tf_gene_weight
        self.align_gene_cfo_weight = align_gene_cfo_weight
        self.align_celltype_tf_weight = align_celltype_tf_weight

        self.contrast_weight = contrast_weight
        skip_types = {"gene"} if contrast_skip_gene else set()
        self.contrast_loss_fn = GraphContrastiveLoss(
            temperature=contrast_temperature,
            num_negatives=contrast_num_negatives,
            batch_size=contrast_batch_size,
            skip_node_types=skip_types,
        )

        logger.info(
            "Graph loss weights: recon=%s, contrast=%s, alignment=%s",
            recon_weight,
            contrast_weight,
            cross_type_align_weight,
        )

    @staticmethod
    def _cosine_target_loss(
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        similarities = (
            F.normalize(src_embeddings, p=2, dim=1)
            * F.normalize(dst_embeddings, p=2, dim=1)
        ).sum(dim=1)
        return F.mse_loss(similarities, target)

    def compute_cross_type_alignment_loss(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weights: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Align the four directly trained relations in the shared embedding space."""
        losses: Dict[str, torch.Tensor] = {}

        if CELLTYPE_CFO in edge_indices and CELLTYPE_CFO in edge_weights:
            edges = edge_indices[CELLTYPE_CFO]
            losses["align_celltype_cfo"] = self._cosine_target_loss(
                h_dict["celltype"][edges[0]],
                h_dict["cfo"][edges[1]],
                edge_weights[CELLTYPE_CFO],
            )

        if TF_GENE in edge_indices:
            edges = edge_indices[TF_GENE]
            target = torch.full_like(edges[0], 0.8, dtype=h_dict["tf"].dtype)
            losses["align_tf_gene"] = self._cosine_target_loss(
                h_dict["tf"][edges[0]],
                h_dict["gene"][edges[1]],
                target,
            )

        if GENE_CFO in edge_indices:
            edges = edge_indices[GENE_CFO]
            target = torch.full_like(edges[0], 0.8, dtype=h_dict["gene"].dtype)
            losses["align_gene_cfo"] = self._cosine_target_loss(
                h_dict["gene"][edges[0]],
                h_dict["cfo"][edges[1]],
                target,
            )

        if CELLTYPE_TF in edge_indices and CELLTYPE_TF in edge_weights:
            edges = edge_indices[CELLTYPE_TF]
            losses["align_celltype_tf"] = self._cosine_target_loss(
                h_dict["celltype"][edges[0]],
                h_dict["tf"][edges[1]],
                edge_weights[CELLTYPE_TF],
            )

        return losses

    @staticmethod
    def compute_reconstruction_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        edge_type: str = "binary",
    ) -> torch.Tensor:
        if edge_type == "continuous" and edge_weights is not None:
            return F.mse_loss(torch.sigmoid(predictions), edge_weights, reduction="mean")
        return F.binary_cross_entropy_with_logits(predictions, targets, reduction="mean")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        edge_weights: Optional[Dict[str, torch.Tensor]] = None,
        contrast_embeddings: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | float = 0.0
        tasks = (
            ("tf_gene_recon", "tf_gene_labels", self.tf_gene_recon_weight, "binary"),
            ("gene_cfo_recon", "gene_cfo_labels", self.gene_cfo_recon_weight, "binary"),
            (
                "celltype_tf_recon",
                "celltype_tf_labels",
                self.celltype_tf_recon_weight,
                "continuous",
            ),
            (
                "celltype_cfo_recon",
                "celltype_cfo_labels",
                self.celltype_cfo_recon_weight,
                "continuous",
            ),
        )

        reconstruction_total: torch.Tensor | float = 0.0
        for prediction_key, target_key, weight, edge_type in tasks:
            if prediction_key not in predictions or target_key not in targets:
                continue
            relation_weights = edge_weights.get(prediction_key) if edge_weights else None
            relation_loss = self.compute_reconstruction_loss(
                predictions[prediction_key],
                targets[target_key],
                relation_weights,
                edge_type,
            )
            losses[prediction_key] = relation_loss
            reconstruction_total = reconstruction_total + weight * relation_loss
        total_loss = total_loss + self.recon_weight * reconstruction_total

        if contrast_embeddings is not None:
            contrast_loss = self.contrast_loss_fn(*contrast_embeddings)
            losses["contrast_loss"] = contrast_loss
            total_loss = total_loss + self.contrast_weight * contrast_loss

        losses["total_loss"] = total_loss
        return losses


class GraphContrastiveLoss(nn.Module):
    """InfoNCE over two independently edge-masked graph views."""

    def __init__(
        self,
        temperature: float = 0.07,
        num_negatives: int = 4096,
        batch_size: int = 2048,
        skip_node_types: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.skip_node_types = skip_node_types or set()

    def forward(
        self,
        h_dict_view1: Dict[str, torch.Tensor],
        h_dict_view2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss: torch.Tensor | float = 0.0
        total_nodes = 0
        for node_type, embeddings1 in h_dict_view1.items():
            if node_type not in h_dict_view2 or node_type in self.skip_node_types:
                continue
            embeddings2 = h_dict_view2[node_type]
            num_nodes = embeddings1.shape[0]
            if num_nodes == 0:
                continue

            normalized1 = F.normalize(embeddings1, p=2, dim=1)
            normalized2 = F.normalize(embeddings2, p=2, dim=1)
            if num_nodes <= self.batch_size:
                loss = self._compute_full_contrastive(normalized1, normalized2)
            else:
                loss = self._compute_sampled_contrastive(
                    normalized1, normalized2, num_nodes
                )
            total_loss = total_loss + loss * num_nodes
            total_nodes += num_nodes

        if total_nodes:
            return total_loss / total_nodes
        return torch.tensor(0.0, device=next(iter(h_dict_view1.values())).device)

    def _compute_full_contrastive(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        similarities = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
        labels = torch.arange(embeddings1.shape[0], device=embeddings1.device)
        return F.cross_entropy(similarities, labels)

    def _compute_sampled_contrastive(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        batch_losses = []
        for start in range(0, num_nodes, self.batch_size):
            end = min(start + self.batch_size, num_nodes)
            batch1 = embeddings1[start:end]
            batch2 = embeddings2[start:end]
            positive = torch.sum(batch1 * batch2, dim=1) / self.temperature
            num_negatives = min(self.num_negatives, num_nodes - 1)
            if num_negatives > 0:
                negative_indices = torch.randint(
                    0, num_nodes, (num_negatives,), device=embeddings1.device
                )
                negative = (
                    torch.matmul(batch1, embeddings2[negative_indices].t())
                    / self.temperature
                )
                logits = torch.cat([positive.unsqueeze(1), negative], dim=1)
                labels = torch.zeros(end - start, dtype=torch.long, device=embeddings1.device)
                batch_losses.append(F.cross_entropy(logits, labels))
            else:
                batch_losses.append(F.mse_loss(batch1, batch2))
        return torch.stack(batch_losses).mean()


def create_loss_function(config: Dict) -> HeteroLoss:
    """Create the graph loss from the ``loss`` config section."""
    return HeteroLoss(**config)
