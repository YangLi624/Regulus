"""Unified perturbation model used by training, prediction and explanation."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from regulus.perturb.models import (
    GeneTokenTransformerEncoder,
    GraphBiasConfig,
    JointCrossTransformerEncoder,
    MLPHead,
    PrototypeMatchingHead,
)
from regulus.perturb.spec import normalize_channels, normalize_head, normalize_mode


class RegulusPerturbationModel(nn.Module):
    """One model surface with channel-specific encoder routing."""

    def __init__(
        self,
        *,
        channels: str,
        head: str,
        h_gene: torch.Tensor,
        h_cfo: Optional[torch.Tensor],
        d_cond: int,
        n_candidates: int,
        h_prototypes: Optional[torch.Tensor] = None,
        gene_topk: int = 2048,
        cfo_topk: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        gene_n_heads: int = 8,
        use_graph_bias: bool = False,
        graph_bias_type: str = "none",
        graph_bias_strength: float = 1.0,
        use_bilinear: bool = True,
        learnable_temperature: bool = True,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = normalize_channels(channels)
        self.head_type = normalize_head(head)
        self.d_cond = int(d_cond)
        self.register_buffer("h_gene", h_gene.detach().clone(), persistent=False)
        self.register_buffer(
            "h_cfo",
            None if h_cfo is None else h_cfo.detach().clone(),
            persistent=False,
        )

        d_embedding = int(h_gene.shape[1])
        self.gene_encoder: Optional[GeneTokenTransformerEncoder] = None
        self.joint_encoder: Optional[JointCrossTransformerEncoder] = None
        if self.channels == "gene":
            self.gene_encoder = GeneTokenTransformerEncoder(
                d_embedding=d_embedding,
                d_cond=self.d_cond,
                topk=gene_topk,
                n_layers=n_layers,
                n_heads=gene_n_heads,
                dropout=dropout,
                graph_bias=GraphBiasConfig(
                    use_graph_bias=use_graph_bias,
                    graph_bias_type=graph_bias_type,
                    graph_bias_strength=graph_bias_strength,
                ),
            )
        else:
            if self.h_cfo is None:
                raise ValueError(f"channels={self.channels!r} requires CFO embeddings")
            self.joint_encoder = JointCrossTransformerEncoder(
                d_embedding=d_embedding,
                d_cond=self.d_cond,
                gene_topk=gene_topk,
                cfo_topk=cfo_topk,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
            )

        if self.head_type == "mlp":
            self.head = MLPHead(d_cond=self.d_cond, n_candidates=n_candidates)
        else:
            if h_prototypes is None:
                raise ValueError("prototype_matching requires h_prototypes")
            if len(h_prototypes) != n_candidates:
                raise ValueError("candidate and prototype counts differ")
            self.head = PrototypeMatchingHead(
                d_cond=self.d_cond,
                h_prototypes=h_prototypes,
                use_bilinear=use_bilinear,
                learnable_temperature=learnable_temperature,
                temperature_init=temperature_init,
            )

    def encode(
        self,
        *,
        gene_input: Optional[torch.Tensor],
        cfo_input: Optional[torch.Tensor],
        mode: Optional[str] = None,
        graph_bias_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_explain: bool = False,
        return_token_embeddings: bool = False,
    ):
        mode = normalize_mode(mode, self.channels)
        if self.gene_encoder is not None:
            if gene_input is None:
                raise ValueError("gene_input is required")
            return self.gene_encoder(
                gene_input,
                self.h_gene,
                graph_bias_fn=graph_bias_fn,
                return_explain=return_explain,
                return_token_embeddings=return_token_embeddings,
            )

        assert self.joint_encoder is not None
        return self.joint_encoder(
            gene_values=gene_input if mode != "cfo_only" else None,
            h_gene=self.h_gene if mode != "cfo_only" else None,
            cfo_values=cfo_input if mode != "gene_only" else None,
            h_cfo=self.h_cfo if mode != "gene_only" else None,
            mode=mode,
            return_explain=return_explain,
            return_token_embeddings=return_token_embeddings,
        )

    def forward(self, *, gene_input=None, cfo_input=None, mode=None, graph_bias_fn=None):
        condition = self.encode(
            gene_input=gene_input,
            cfo_input=cfo_input,
            mode=mode,
            graph_bias_fn=graph_bias_fn,
        )
        return self.head(condition)
