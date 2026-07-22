"""Self-contained prediction, explanation and virtual CFO manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from regulus.graph.frozen_hgt import FrozenHGTBundle
from regulus.io.bundle import BundleManifest, load_bundle_manifest
from regulus.io.h5ad_input import build_predict_dataset
from regulus.io.obs_filter import subset_dataset
from regulus.perturb.model import RegulusPerturbationModel
from regulus.perturb.prototype_utils import build_candidate_prototypes
from regulus.perturb.spec import normalize_mode
from regulus.utils.amp import autocast_context


class RegulusModel:
    """Loaded perturbation checkpoint independent of its training H5AD."""

    def __init__(self, manifest: BundleManifest, device: Optional[str] = None) -> None:
        self.manifest = manifest
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_representation = manifest.representation
        self.channels = manifest.channels

        checkpoint = torch.load(manifest.perturb_ckpt, map_location=self.device, weights_only=False)
        self.candidate_perturbations = list(checkpoint.get("candidate_perturbations", []))
        if not self.candidate_perturbations:
            raise ValueError("perturbation checkpoint does not contain candidate_perturbations")
        self.candidate_is_tf = dict(checkpoint.get("candidate_is_tf", {}))

        with manifest.train_config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model_config = config["model"]
        graph_kwargs = {
            "model_path": str(manifest.graph_ckpt),
            "device": str(self.device),
        }
        graph_asset_dir = manifest.graph_asset_dir or config.get("data", {}).get("graph_asset_dir")
        if graph_asset_dir is not None:
            graph_kwargs["graph_asset_dir"] = str(graph_asset_dir)
        self._hgt = FrozenHGTBundle(**graph_kwargs).load()
        assert self._hgt.node_embeddings is not None
        h_gene = self._hgt.node_embeddings["gene"].to(self.device)
        h_cfo = self._hgt.node_embeddings.get("cfo")
        if h_cfo is not None:
            h_cfo = h_cfo.to(self.device)

        prototypes = None
        if manifest.head == "prototype_matching":
            prototypes, _ = build_candidate_prototypes(
                self.candidate_perturbations,
                self.candidate_is_tf,
                self._hgt,
                manifest.gene_universe,
                device=self.device,
            )
        layers = int(model_config.get("layers", 2))
        heads = int(model_config.get("heads", 8 if self.channels == "gene" else 4))
        dropout = float(model_config.get("dropout", 0.1))

        self.model = RegulusPerturbationModel(
            channels=self.channels,
            head=manifest.head,
            h_gene=h_gene,
            h_cfo=h_cfo,
            d_cond=int(model_config.get("d_cond", 128)),
            n_candidates=len(self.candidate_perturbations),
            h_prototypes=prototypes,
            gene_topk=int(model_config.get("gene_topk", 2048)),
            cfo_topk=int(model_config.get("cfo_topk", 256)),
            n_layers=layers,
            n_heads=heads,
            gene_n_heads=heads,
            dropout=dropout,
            use_graph_bias=bool(model_config.get("use_graph_bias", False)),
            graph_bias_type=str(model_config.get("graph_bias_type", "none")),
            graph_bias_strength=float(model_config.get("graph_bias_strength", 1.0)),
            use_bilinear=bool(model_config.get("use_bilinear", True)),
            learnable_temperature=bool(model_config.get("learnable_temperature", True)),
            temperature_init=float(model_config.get("temperature_init", 1.0)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.mixed_precision = bool(config.get("training", {}).get("mixed_precision", True))

    @classmethod
    def from_bundle(
        cls,
        bundle_path: Union[str, Path],
        *,
        input_representation: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "RegulusModel":
        manifest = load_bundle_manifest(bundle_path)
        manifest.validate_input_representation(input_representation)
        return cls(manifest, device=device)

    def _dataset(self, path: Union[str, Path]):
        expected_cfo_ids = None
        if self.channels in ("cfo", "gene_cfo") and self._hgt.cfo_nodes is not None:
            cfo_nodes = self._hgt.cfo_nodes
            if "cfo_id" in cfo_nodes:
                if "cfo_index" in cfo_nodes:
                    cfo_nodes = cfo_nodes.sort_values("cfo_index")
                expected_cfo_ids = cfo_nodes["cfo_id"].astype(str).tolist()
        return build_predict_dataset(
            path,
            self.manifest.gene_universe,
            representation=self.input_representation,
            channels=self.channels,
            expected_cfo_ids=expected_cfo_ids,
        )

    def _sample_inputs(self, sample: dict):
        gene = sample.get("gene_input")
        cfo = sample.get("cfo_input")
        if gene is not None:
            gene = gene.to(self.device)
            if gene.ndim == 1:
                gene = gene.unsqueeze(0)
        if cfo is not None:
            cfo = cfo.to(self.device)
            if cfo.ndim == 1:
                cfo = cfo.unsqueeze(0)
        return gene, cfo

    def _logits(self, sample: dict, mode: str) -> torch.Tensor:
        gene, cfo = self._sample_inputs(sample)
        return self.model(gene_input=gene, cfo_input=cfo, mode=mode)

    @torch.no_grad()
    def predict(
        self,
        h5ad_path: Union[str, Path],
        *,
        mode: Optional[str] = None,
        top_k: int = 50,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        dataset = self._dataset(h5ad_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        public_mode = normalize_mode(mode or self.manifest.default_mode, self.channels)
        rows = []
        for batch in loader:
            with autocast_context(
                self.device, enabled=self.mixed_precision and self.device.type == "cuda"
            ):
                logits = self._logits(batch, public_mode)
            k = min(int(top_k), logits.shape[1])
            scores, indices = logits.topk(k=k, dim=1)
            cell_ids = list(batch["cell_id"])
            for i, cell_id in enumerate(cell_ids):
                for rank, (candidate_index, score) in enumerate(
                    zip(indices[i].tolist(), scores[i].tolist()), start=1
                ):
                    rows.append({
                        "cell_id": str(cell_id),
                        "rank": rank,
                        "perturbation": self.candidate_perturbations[int(candidate_index)],
                        "score": float(score),
                        "mode": public_mode,
                    })
        return pd.DataFrame(rows)

    @torch.no_grad()
    def manipulate(
        self,
        *,
        cfo_targets: Sequence[str],
        cfo_delta: Sequence[float],
        anchor_h5ad: Union[str, Path],
        obs_filter: Optional[str] = None,
        mode: str = "cfo_only",
        edit_mode: str = "anchor_plus_delta",
        top_k: int = 20,
        sort_by: str = "rank_gain",
    ) -> pd.DataFrame:
        from regulus.manipulate.core import run_manipulation

        return run_manipulation(
            self,
            cfo_targets=cfo_targets,
            cfo_delta=cfo_delta,
            anchor_h5ad=anchor_h5ad,
            obs_filter=obs_filter,
            mode=mode,
            edit_mode=edit_mode,
            top_k=top_k,
            sort_by=sort_by,
        )

    def explain(
        self,
        h5ad_path: Union[str, Path],
        *,
        mode: Optional[str] = None,
        output_dir: Union[str, Path] = "explain_out",
        attribution_channel: Optional[str] = None,
        attribution_method: str = "gradient_x_input",
        target_candidate: Optional[str] = None,
        ig_steps: int = 16,
        top_k_features: int = 20,
        top_k_candidates: int = 1,
        render_pdf: bool = False,
        obs_filter: Optional[str] = None,
    ):
        from regulus.explain.pipeline import run_explain

        dataset = self._dataset(h5ad_path)
        subset, _, _ = subset_dataset(dataset, obs_filter)
        runtime_mode = normalize_mode(mode or self.manifest.default_mode, self.channels)
        resolved_channel = attribution_channel or (
            "cfo" if runtime_mode == "cfo_only" else "gene"
        )
        return run_explain(
            self,
            subset,
            output_dir,
            mode=runtime_mode,
            attribution_channel=resolved_channel,
            attribution_method=attribution_method,
            target_candidate=target_candidate,
            ig_steps=ig_steps,
            top_k_features=top_k_features,
            top_k_candidates=top_k_candidates,
            render_pdf=render_pdf,
        )
