"""Training loop for Regulus heterogeneous graph pretraining."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch_geometric.data import HeteroData

from regulus.graph.model import load_graph_state_dict
from regulus.graph.schema import CELLTYPE_CFO, CELLTYPE_TF, GENE_CFO, TF_GENE, TRAINED_EDGE_TYPES

logger = logging.getLogger(__name__)


class Trainer:
    """Full-batch HGT trainer with masked-edge reconstruction."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        use_mixed_precision: bool = True,
        patience: int = 10,
        min_delta: float = 1e-4,
        save_dir: str = "outputs/checkpoints",
        edge_mask_ratio: float = 0.1,
        recon_mask_ratio: float = 0.3,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.use_mixed_precision = use_mixed_precision
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir
        self.edge_mask_ratio = edge_mask_ratio
        self.recon_mask_ratio = recon_mask_ratio
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self.scaler = GradScaler() if use_mixed_precision else None

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_align_loss = float("inf")
        self.patience_counter = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_recon_losses: List[float] = []
        self.val_recon_losses: List[float] = []
        self.train_align_losses: List[float] = []
        self.val_align_losses: List[float] = []
        self.train_tf_gene_losses: List[float] = []
        self.train_gene_go_losses: List[float] = []
        self.train_celltype_tf_losses: List[float] = []
        self.train_celltype_go_losses: List[float] = []
        self.val_tf_gene_losses: List[float] = []
        self.val_gene_go_losses: List[float] = []
        self.val_celltype_tf_losses: List[float] = []
        self.val_celltype_go_losses: List[float] = []

        logger.info(
            "Initialized Trainer with lr=%s, mixed_precision=%s, patience=%s",
            learning_rate,
            use_mixed_precision,
            patience,
        )

    def _masked_edges(
        self,
        full_edges: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_edges = full_edges.shape[1]
        num_mask = int(num_edges * max(0.0, min(1.0, self.recon_mask_ratio)))
        if num_mask <= 0:
            return full_edges, full_edges, edge_weights
        permutation = torch.randperm(num_edges, device=full_edges.device)
        mask_indices = permutation[:num_mask]
        keep_indices = permutation[num_mask:]
        encoder_edges = full_edges[:, keep_indices] if keep_indices.numel() else full_edges
        masked_weights = edge_weights[mask_indices] if edge_weights is not None else None
        return full_edges[:, mask_indices], encoder_edges, masked_weights

    def prepare_batch_data(self, data: HeteroData) -> Tuple[Dict, Dict]:
        """Attach reconstruction samples for the four directly trained relations."""
        targets: Dict[str, torch.Tensor] = {}
        edge_weights: Dict[str, torch.Tensor] = {}

        binary_tasks = (
            (TF_GENE, "tf_gene_labels", "tf", "gene"),
            (GENE_CFO, "gene_go_labels", "gene", "go"),
        )
        for edge_type, target_key, src_type, dst_type in binary_tasks:
            if edge_type not in data.edge_types:
                continue
            full_edges = data[edge_type].edge_index
            masked_edges, encoder_edges, _ = self._masked_edges(full_edges)
            num_positive = masked_edges.shape[1]
            negative_edges = self._negative_sample_edges(
                data[src_type].x.shape[0],
                data[dst_type].x.shape[0],
                full_edges,
                num_positive // 2,
            )
            data[edge_type].edge_index_encoder = encoder_edges
            data[edge_type].recon_edge_index = torch.cat(
                [masked_edges, negative_edges], dim=1
            )
            targets[target_key] = torch.cat(
                [
                    torch.ones(num_positive, device=self.device),
                    torch.zeros(negative_edges.shape[1], device=self.device),
                ]
            )

        continuous_tasks = (
            (CELLTYPE_TF, "celltype_tf_recon", "celltype_tf_labels"),
            (CELLTYPE_CFO, "celltype_go_recon", "celltype_go_labels"),
        )
        for edge_type, prediction_key, target_key in continuous_tasks:
            if edge_type not in data.edge_types:
                continue
            full_edges = data[edge_type].edge_index
            masked_edges, encoder_edges, masked_weights = self._masked_edges(
                full_edges, data[edge_type].edge_attr
            )
            data[edge_type].edge_index_encoder = encoder_edges
            data[edge_type].recon_edge_index = masked_edges
            edge_weights[prediction_key] = masked_weights
            targets[target_key] = masked_weights

        return targets, edge_weights

    @staticmethod
    def _negative_sample_edges(
        num_src: int,
        num_dst: int,
        pos_edges: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample absent edges using the paper-training rejection sampler."""
        device = pos_edges.device
        if num_samples <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        total_possible = num_src * num_dst
        num_samples = min(num_samples, total_possible - pos_edges.shape[1])
        if num_samples <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        if total_possible < 100_000_000:
            positives = set((pos_edges[0] * num_dst + pos_edges[1]).cpu().tolist())
            use_flat_indices = True
        else:
            positives = set(
                zip(pos_edges[0].cpu().tolist(), pos_edges[1].cpu().tolist())
            )
            use_flat_indices = False

        negatives = []
        max_iterations = max(num_samples * 50, 10_000_000)
        iterations = 0
        while len(negatives) < num_samples and iterations < max_iterations:
            remaining = num_samples - len(negatives)
            batch_size = min(remaining * 10, 50_000)
            src = torch.randint(0, num_src, (batch_size,), device=device)
            dst = torch.randint(0, num_dst, (batch_size,), device=device)
            if use_flat_indices:
                for candidate in (src * num_dst + dst).cpu().tolist():
                    if candidate not in positives:
                        negatives.append([candidate // num_dst, candidate % num_dst])
                        if len(negatives) >= num_samples:
                            break
            else:
                for src_index, dst_index in zip(src.cpu().tolist(), dst.cpu().tolist()):
                    if (src_index, dst_index) not in positives:
                        negatives.append([src_index, dst_index])
                        if len(negatives) >= num_samples:
                            break
            iterations += batch_size

        if not negatives:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        if len(negatives) < num_samples:
            logger.warning(
                "Negative sampler produced %d/%d requested edges",
                len(negatives),
                num_samples,
            )
        return torch.tensor(
            negatives[:num_samples], dtype=torch.long, device=device
        ).t()

    def _graph_inputs(self, data: HeteroData) -> Tuple[Dict, Dict]:
        x_dict = {
            node_type: data[node_type].x
            for node_type in self.model.node_types
            if node_type in data.node_types
        }
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.model.edge_types
            if edge_type in data.edge_types
        }
        return x_dict, edge_index_dict

    def _base_forward(self, data: HeteroData) -> Tuple[Dict, Dict, torch.Tensor]:
        targets, edge_weights = self.prepare_batch_data(data)
        contrast_embeddings = None
        if self.loss_fn.contrast_weight > 0:
            x_dict, edge_index_dict = self._graph_inputs(data)
            view1 = self.model.generate_edge_mask(edge_index_dict, self.edge_mask_ratio)
            view2 = self.model.generate_edge_mask(edge_index_dict, self.edge_mask_ratio)
        predictions = self.model(data)
        if self.loss_fn.contrast_weight > 0:
            contrast_embeddings = (
                self.model.encode(x_dict, edge_index_dict, view1),
                self.model.encode(x_dict, edge_index_dict, view2),
            )
        losses = self.loss_fn(
            predictions,
            targets,
            edge_weights,
            contrast_embeddings=contrast_embeddings,
        )
        return predictions, losses, losses["total_loss"]

    def _alignment_loss(
        self, data: HeteroData, predictions: Dict
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        if self.loss_fn.cross_type_align_weight <= 0:
            return {}, zero
        edge_indices = {
            edge_type: data[edge_type].edge_index
            for edge_type in TRAINED_EDGE_TYPES
            if edge_type in data.edge_types
        }
        edge_weights = {
            edge_type: data[edge_type].edge_attr
            for edge_type in TRAINED_EDGE_TYPES
            if edge_type in data.edge_types
            and hasattr(data[edge_type], "edge_attr")
            and data[edge_type].edge_attr is not None
        }
        losses = self.loss_fn.compute_cross_type_alignment_loss(
            predictions["node_embeddings"], edge_indices, edge_weights
        )
        weighted = (
            losses.get("align_celltype_go", zero) * self.loss_fn.align_celltype_go_weight
            + losses.get("align_tf_gene", zero) * self.loss_fn.align_tf_gene_weight
            + losses.get("align_gene_go", zero) * self.loss_fn.align_gene_go_weight
            + losses.get("align_celltype_tf", zero) * self.loss_fn.align_celltype_tf_weight
        )
        return losses, weighted

    def train_epoch(self, data: HeteroData) -> Dict[str, float]:
        self.model.train()
        data = data.to(self.device)
        if self.use_mixed_precision:
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                predictions, losses, total_loss = self._base_forward(data)
                align_losses, align_total = self._alignment_loss(data, predictions)
                total_loss = total_loss + self.loss_fn.cross_type_align_weight * align_total
        else:
            predictions, losses, total_loss = self._base_forward(data)
            align_losses, align_total = self._alignment_loss(data, predictions)
            total_loss = total_loss + self.loss_fn.cross_type_align_weight * align_total

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        metrics = {f"train_{key}": value.item() for key, value in losses.items()}
        metrics.update(
            {f"train_{key}": value.item() for key, value in align_losses.items()}
        )
        metrics["train_cross_type_align_total"] = align_total.item()
        metrics["train_total_loss"] = total_loss.item()
        return metrics

    def validate_epoch(self, data: HeteroData) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            if self.use_mixed_precision:
                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    predictions, losses, total_loss = self._base_forward(data)
                    align_losses, align_total = self._alignment_loss(data, predictions)
                    total_loss = total_loss + self.loss_fn.cross_type_align_weight * align_total
            else:
                predictions, losses, total_loss = self._base_forward(data)
                align_losses, align_total = self._alignment_loss(data, predictions)
                total_loss = total_loss + self.loss_fn.cross_type_align_weight * align_total

        metrics = {f"val_{key}": value.item() for key, value in losses.items()}
        metrics.update({f"val_{key}": value.item() for key, value in align_losses.items()})
        metrics["val_cross_type_align_total"] = align_total.item()
        metrics["val_total_loss"] = total_loss.item()
        return metrics

    def early_stopping_check(self, val_loss: float, val_metrics: Dict[str, float]) -> bool:
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.patience_counter += 1
            return self.patience_counter >= self.patience
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            align_loss = val_metrics.get("val_cross_type_align_total", float("inf"))
            if not (np.isnan(align_loss) or np.isinf(align_loss)):
                self.best_val_align_loss = min(self.best_val_align_loss, align_loss)
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def _history(self) -> Dict[str, List[float] | float | int]:
        return {
            "train_total_loss": self.train_losses,
            "val_total_loss": self.val_losses,
            "train_recon_loss": self.train_recon_losses,
            "val_recon_loss": self.val_recon_losses,
            "train_align_loss": self.train_align_losses,
            "val_align_loss": self.val_align_losses,
            "train_tf_gene_loss": self.train_tf_gene_losses,
            "train_gene_go_loss": self.train_gene_go_losses,
            "train_celltype_tf_loss": self.train_celltype_tf_losses,
            "train_celltype_go_loss": self.train_celltype_go_losses,
            "val_tf_gene_loss": self.val_tf_gene_losses,
            "val_gene_go_loss": self.val_gene_go_losses,
            "val_celltype_tf_loss": self.val_celltype_tf_losses,
            "val_celltype_go_loss": self.val_celltype_go_losses,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_align_loss": self.best_val_align_loss,
            "metrics": metrics,
            "history": self._history(),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        load_graph_state_dict(self.model, checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except ValueError:
            logger.warning(
                "Loaded graph weights but not the paper-era optimizer state because retired decoder parameters changed its shape"
            )
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", -1) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_align_loss = checkpoint.get(
            "best_val_align_loss", float("inf")
        )
        history = checkpoint.get("history", checkpoint)
        self.train_losses = history.get("train_total_loss", history.get("train_losses", []))
        self.val_losses = history.get("val_total_loss", history.get("val_losses", []))

    def save_training_history(self, save_path: str) -> None:
        history = self._history()
        history["epochs"] = list(range(1, len(self.train_losses) + 1))
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    @staticmethod
    def _reconstruction_total(metrics: Dict[str, float], prefix: str) -> float:
        return sum(
            metrics.get(f"{prefix}_{key}_recon", 0.0)
            for key in ("tf_gene", "gene_go", "celltype_tf", "celltype_go")
        )

    def fit(
        self,
        train_data: HeteroData,
        val_data: HeteroData,
        epochs: int = 100,
        log_interval: int = 10,
    ) -> Dict:
        start_time = time.time()
        for epoch in range(self.current_epoch, epochs):
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_data)
            val_metrics = self.validate_epoch(val_data)
            val_loss = val_metrics["val_total_loss"]
            self.scheduler.step(
                train_metrics["train_total_loss"]
                if np.isnan(val_loss) or np.isinf(val_loss)
                else val_loss
            )

            self.train_losses.append(train_metrics["train_total_loss"])
            self.val_losses.append(val_loss)
            train_relations = (
                self.train_tf_gene_losses,
                self.train_gene_go_losses,
                self.train_celltype_tf_losses,
                self.train_celltype_go_losses,
            )
            val_relations = (
                self.val_tf_gene_losses,
                self.val_gene_go_losses,
                self.val_celltype_tf_losses,
                self.val_celltype_go_losses,
            )
            relation_keys = ("tf_gene", "gene_go", "celltype_tf", "celltype_go")
            for values, key in zip(train_relations, relation_keys):
                values.append(train_metrics.get(f"train_{key}_recon", 0.0))
            for values, key in zip(val_relations, relation_keys):
                values.append(val_metrics.get(f"val_{key}_recon", 0.0))
            self.train_recon_losses.append(self._reconstruction_total(train_metrics, "train"))
            self.val_recon_losses.append(self._reconstruction_total(val_metrics, "val"))
            self.train_align_losses.append(
                train_metrics.get("train_cross_type_align_total", 0.0)
            )
            self.val_align_losses.append(
                val_metrics.get("val_cross_type_align_total", 0.0)
            )

            is_best = (
                not np.isnan(val_loss)
                and val_loss < self.best_val_loss - self.min_delta
            )
            should_stop = self.early_stopping_check(val_loss, val_metrics)
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            if epoch % log_interval == 0:
                logger.info(
                    "Epoch %d | train %.4f | val %.4f | lr %.2e | %.2fs",
                    epoch,
                    train_metrics["train_total_loss"],
                    val_loss,
                    self.optimizer.param_groups[0]["lr"],
                    time.time() - epoch_start,
                )
            self.current_epoch = epoch + 1
            if should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        logger.info("Training completed in %.2fs", time.time() - start_time)
        return self._history()

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        fig, axis = plt.subplots(figsize=(5, 4))
        epochs = range(1, len(self.train_losses) + 1)
        axis.plot(epochs, self.train_losses, label="Train")
        axis.plot(epochs, self.val_losses, label="Validation")
        axis.set(xlabel="Epoch", ylabel="Loss", title="Graph pretraining")
        axis.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
