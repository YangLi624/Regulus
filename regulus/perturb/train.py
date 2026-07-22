"""Training loop for the released Regulus perturbation model."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from regulus.graph.frozen_hgt import FrozenHGTBundle
from regulus.perturb.data import PerturbationDataset
from regulus.perturb.model import RegulusPerturbationModel
from regulus.perturb.prototype_utils import build_candidate_prototypes
from regulus.perturb.spec import PerturbModelSpec, normalize_mode
from regulus.perturb.training_modes import (
    aggregate_metric_lists,
    compute_topk_accuracy,
    normalize_mode_schedule,
    sample_encode_mode,
)
from regulus.utils.amp import autocast_context, make_grad_scaler

logger = logging.getLogger(__name__)


def _channel_schedule(raw: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    return normalize_mode_schedule(raw)


class PerturbTrainer:
    """Train one Regulus architecture with cross-entropy supervision."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        inference_only: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.spec = PerturbModelSpec.from_config(self.config)
        self.channels = self.spec.channels
        self.representation = self.spec.representation
        self.head_type = self.spec.head
        self.inference_only = bool(inference_only)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        data = self.config["data"]
        self.dataset_dir = Path(data["dataset_dir"])
        self.train_data = str(data["train_data"])
        self.test_data = data.get("test_data")
        self.gene_universe_path = str(data["gene_universe"])

        model = self.config["model"]
        self.d_cond = int(model.get("d_cond", 128))
        self.hidden_dim = int(model.get("hidden_dim", 128))
        self.checkpoint_path = str(model["graph_checkpoint"])
        self.gene_topk = int(model.get("gene_topk", 2048))
        self.cfo_topk = int(model.get("cfo_topk", 256))
        if self.channels == "gene":
            self.encoder_layers = int(model.get("layers", 2))
            self.encoder_heads = int(model.get("heads", 8))
            self.encoder_dropout = float(model.get("dropout", 0.1))
        else:
            self.encoder_layers = int(model.get("layers", 2))
            self.encoder_heads = int(model.get("heads", 4))
            self.encoder_dropout = float(model.get("dropout", 0.1))
        self.use_graph_bias = bool(model.get("use_graph_bias", False))
        self.graph_bias_type = str(model.get("graph_bias_type", "none"))
        self.graph_bias_strength = float(model.get("graph_bias_strength", 1.0))
        self.use_bilinear = bool(model.get("use_bilinear", True))
        self.learnable_temperature = bool(model.get("learnable_temperature", True))
        self.temperature_init = float(model.get("temperature_init", 1.0))

        training = self.config["training"]
        loss_name = str(training.get("loss", "cross_entropy"))
        if loss_name != "cross_entropy":
            raise ValueError("The released perturbation trainer supports cross_entropy only")
        self.learning_rate = float(training["learning_rate"])
        self.batch_size = int(training["batch_size"])
        self.epochs = int(training["epochs"])
        self.mixed_precision = bool(training.get("mixed_precision", True))
        self.grad_clip_norm = float(training.get("grad_clip_norm", 1.0))
        self.checkpoint_interval = int(training.get("checkpoint_interval", 5000))
        self.log_interval = int(training.get("log_interval", 10))
        self.random_seed = int(training.get("random_seed", 42))
        self.mode_schedule = _channel_schedule(training.get("channel_schedule"))
        if self.channels == "gene_cfo" and self.mode_schedule is None:
            raise ValueError("gene_cfo training requires training.channel_schedule")
        self._mode_rng = random.Random(self.random_seed)

        scheduler = training.get("lr_scheduler", {}) or {}
        self.lr_scheduler_type = scheduler.get("type")
        self.lr_warmup_steps = int(scheduler.get("warmup_steps", 0))
        self.lr_min_lr = float(scheduler.get("min_lr", 0.0))

        self.output_dir = Path(self.config.get("output_dir", "outputs/conditional"))
        self.checkpoint_dir = Path(
            self.config.get("checkpoint_dir", self.output_dir / "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resume = bool(self.config.get("resume", True))
        self.step = 0
        self.epoch = 0
        self.best_val_top1_acc = 0.0
        self.best_val_epoch = 0

        self.candidate_perturbations, self.candidate_is_tf = self._extract_candidates()
        self.n_candidates = len(self.candidate_perturbations)
        self.perturb_to_idx = {
            perturbation: i for i, perturbation in enumerate(self.candidate_perturbations)
        }
        self._init_model()

        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = F.cross_entropy
        if not self.inference_only:
            self._init_data_loaders()
            self._init_optimizer()
            if self.resume:
                self.load_checkpoint()

    def _extract_candidates(self) -> tuple[list[str], Dict[str, bool]]:
        train_path = self.dataset_dir / self.train_data
        adata = sc.read_h5ad(train_path, backed="r")
        if "perturbation" not in adata.obs:
            raise ValueError("training h5ad requires obs['perturbation']")
        labels = adata.obs["perturbation"].astype(str)
        candidates = sorted(set(labels[labels != "control"]))
        adata.file.close()

        tf_path = Path(self.gene_universe_path).parent / "tf_universe.csv"
        if tf_path.exists():
            tf_set = set(pd.read_csv(tf_path)["tf_symbol"].astype(str))
            is_tf = {name: name in tf_set for name in candidates}
        else:
            is_tf = {name: False for name in candidates}
        return candidates, is_tf

    def _init_model(self) -> None:
        graph_asset_dir = self.config["data"].get("graph_asset_dir")
        graph_kwargs = {
            "model_path": self.checkpoint_path,
            "device": str(self.device),
        }
        if graph_asset_dir is not None:
            graph_kwargs["graph_asset_dir"] = graph_asset_dir
        self._hgt = FrozenHGTBundle(**graph_kwargs).load()
        assert self._hgt.node_embeddings is not None
        self.h_gene = self._hgt.node_embeddings["gene"].detach().to(self.device)
        self.h_cfo = self._hgt.node_embeddings.get("cfo")
        if self.h_cfo is not None:
            self.h_cfo = self.h_cfo.detach().to(self.device)

        prototypes = None
        self.prototype_kinds = None
        if self.head_type == "prototype_matching":
            prototypes, self.prototype_kinds = build_candidate_prototypes(
                self.candidate_perturbations,
                self.candidate_is_tf,
                self._hgt,
                self.gene_universe_path,
                device=self.device,
            )

        self.model = RegulusPerturbationModel(
            channels=self.channels,
            head=self.head_type,
            h_gene=self.h_gene,
            h_cfo=self.h_cfo,
            d_cond=self.d_cond,
            n_candidates=self.n_candidates,
            h_prototypes=prototypes,
            gene_topk=self.gene_topk,
            cfo_topk=self.cfo_topk,
            n_layers=self.encoder_layers,
            n_heads=self.encoder_heads,
            gene_n_heads=self.encoder_heads,
            dropout=self.encoder_dropout,
            use_graph_bias=self.use_graph_bias,
            graph_bias_type=self.graph_bias_type,
            graph_bias_strength=self.graph_bias_strength,
            use_bilinear=self.use_bilinear,
            learnable_temperature=self.learnable_temperature,
            temperature_init=self.temperature_init,
        ).to(self.device)

    def _make_dataset(self, filename: str) -> PerturbationDataset:
        return PerturbationDataset(
            self.dataset_dir / filename,
            self.gene_universe_path,
            representation=self.representation,
            channels=self.channels,
            candidate_to_index=self.perturb_to_idx,
            require_labels=True,
            expected_n_cfo=(None if self.h_cfo is None else int(self.h_cfo.shape[0])),
        )

    def _init_data_loaders(self) -> None:
        train_dataset = self._make_dataset(self.train_data)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        if self.test_data:
            self.val_loader = DataLoader(
                self._make_dataset(str(self.test_data)),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )

    def _init_optimizer(self) -> None:
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scaler = make_grad_scaler(
            enabled=self.mixed_precision and self.device.type == "cuda"
        )
        if self.lr_scheduler_type == "warmup_cosine":
            steps_per_epoch = max(1, len(self.train_loader))
            total_steps = max(1, self.epochs * steps_per_epoch)
            min_ratio = self.lr_min_lr / self.learning_rate

            def factor(step: int) -> float:
                if self.lr_warmup_steps and step < self.lr_warmup_steps:
                    return max(step, 1) / self.lr_warmup_steps
                progress = (step - self.lr_warmup_steps) / max(
                    1, total_steps - self.lr_warmup_steps
                )
                cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
                return min_ratio + (1 - min_ratio) * cosine

            self.lr_scheduler = LambdaLR(self.optimizer, factor)

    def _select_mode(self) -> str:
        if self.mode_schedule is None:
            return normalize_mode(None, self.channels)
        return sample_encode_mode(self.mode_schedule, self._mode_rng)

    def _batch_inputs(self, batch: dict) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        gene = batch.get("gene_input")
        cfo = batch.get("cfo_input")
        return (
            None if gene is None else gene.to(self.device, non_blocking=True),
            None if cfo is None else cfo.to(self.device, non_blocking=True),
        )

    def _encode_condition(self, batch: dict, mode: Optional[str]):
        gene, cfo = self._batch_inputs(batch)
        return self.model.encode(gene_input=gene, cfo_input=cfo, mode=mode)

    def train_step(self, batch: dict) -> Dict[str, float]:
        assert self.optimizer is not None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        labels = batch["label"].to(self.device)
        valid = labels >= 0
        if not bool(valid.any()):
            raise ValueError("batch contains no candidate labels")
        mode = self._select_mode()
        with autocast_context(self.device, enabled=self.scaler.is_enabled()):
            logits = self.model.head(self._encode_condition(batch, mode))[valid]
            loss = F.cross_entropy(logits, labels[valid])
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_stepped = not self.scaler.is_enabled() or self.scaler.get_scale() >= scale_before
        if self.lr_scheduler is not None and optimizer_stepped:
            self.lr_scheduler.step()
        metrics = compute_topk_accuracy(logits.detach(), labels[valid])
        return {"loss": float(loss.detach()), "mode": mode, **metrics}

    @torch.no_grad()
    def _validate_mode(self, mode: str) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        values: Dict[str, list[float]] = {
            "val_loss": [], "val_top1_acc": [], "val_top5_acc": [], "val_top10_acc": []
        }
        for batch in self.val_loader:
            labels = batch["label"].to(self.device)
            valid = labels >= 0
            if not bool(valid.any()):
                continue
            logits = self.model.head(self._encode_condition(batch, mode))[valid]
            values["val_loss"].append(float(F.cross_entropy(logits, labels[valid])))
            metrics = compute_topk_accuracy(logits, labels[valid])
            for key, value in metrics.items():
                values[f"val_{key}"].append(value)
        return aggregate_metric_lists(values)

    def validate(self) -> Dict[str, float]:
        if self.channels != "gene_cfo":
            return self._validate_mode(normalize_mode(None, self.channels))
        by_mode = {mode: self._validate_mode(mode) for mode in ("joint", "gene_only", "cfo_only")}
        result = dict(by_mode["joint"])
        for mode, metrics in by_mode.items():
            result.update({f"{mode}_{key}": value for key, value in metrics.items()})
        return result

    def train(self) -> None:
        if self.train_loader is None:
            raise RuntimeError("trainer was created for inference only")
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            recent: list[Dict[str, float]] = []
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                recent.append(metrics)
                self.step += 1
                if self.step % self.log_interval == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f top1=%.4f mode=%s",
                        epoch,
                        self.step,
                        np.mean([item["loss"] for item in recent[-self.log_interval:]]),
                        np.mean([item["top1_acc"] for item in recent[-self.log_interval:]]),
                        metrics["mode"],
                    )
                if self.step % self.checkpoint_interval == 0:
                    self.save_checkpoint(archive=True)
            validation = self.validate()
            top1 = float(validation.get("val_top1_acc", 0.0))
            if top1 > self.best_val_top1_acc:
                self.best_val_top1_acc = top1
                self.best_val_epoch = epoch
                self.save_checkpoint(is_best=True)
            self.save_checkpoint()
        if self.val_loader is None:
            self.best_val_epoch = self.epoch
            self.save_checkpoint(is_best=True)

    def _checkpoint_payload(self) -> dict:
        payload = {
            "checkpoint_version": 1,
            "step": self.step,
            "epoch": self.epoch,
            "model_spec": self.spec.as_dict(),
            "candidate_perturbations": self.candidate_perturbations,
            "candidate_is_tf": self.candidate_is_tf,
            "prototype_kinds": self.prototype_kinds,
            "n_candidates": self.n_candidates,
            "channel_schedule": self.mode_schedule,
            "supported_modes": ["joint", "gene_only", "cfo_only"] if self.channels == "gene_cfo" else [normalize_mode(None, self.channels)],
            "config": self.config,
            "validation_available": self.val_loader is not None,
            "best_val_top1_acc": self.best_val_top1_acc,
            "best_val_epoch": self.best_val_epoch,
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            payload["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        return payload

    def save_checkpoint(self, is_best: bool = False, archive: bool = False) -> None:
        payload = self._checkpoint_payload()
        if archive:
            torch.save(payload, self.checkpoint_dir / f"checkpoint_step_{self.step}.pt")
        torch.save(payload, self.checkpoint_dir / "latest_checkpoint.pt")
        if is_best:
            torch.save(payload, self.checkpoint_dir / "best_model.pt")

    def load_checkpoint_from_path(self, ckpt_path: Union[str, Path]) -> None:
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        saved_candidates = checkpoint.get("candidate_perturbations")
        if saved_candidates is not None and list(saved_candidates) != self.candidate_perturbations:
            raise ValueError(
                "checkpoint candidate order does not match the configured training data"
            )
        self.step = int(checkpoint.get("step", 0))
        self.epoch = int(checkpoint.get("epoch", 0))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.best_val_top1_acc = float(checkpoint.get("best_val_top1_acc", 0.0))
        self.best_val_epoch = int(checkpoint.get("best_val_epoch", 0))

    def load_checkpoint(self) -> None:
        latest = self.checkpoint_dir / "latest_checkpoint.pt"
        if not latest.exists():
            return
        self.load_checkpoint_from_path(latest)
        checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
        if self.optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
