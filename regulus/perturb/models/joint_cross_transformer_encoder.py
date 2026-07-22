"""Joint transformer encoder for gene and CFO input streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JointCrossTransformerConfig:
    d_embedding: int
    d_cond: int
    gene_topk: int = 2048
    cfo_topk: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1


class _MultiheadAttention(nn.Module):
    """Memory-efficient multi-head attention backed by PyTorch SDPA."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]
        if key.shape[:1] != (batch_size,) or value.shape[:2] != (batch_size, key_length):
            raise ValueError("Attention inputs have incompatible batch or sequence dimensions")

        def split_heads(values: torch.Tensor, length: int) -> torch.Tensor:
            return values.view(batch_size, length, self.n_heads, self.d_head).transpose(1, 2)

        query = split_heads(self.q_proj(query), query_length)
        key = split_heads(self.k_proj(key), key_length)
        value = split_heads(self.v_proj(value), key_length)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.d_model
        )
        return self.dropout(self.out_proj(output))


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _GeneCFOCrossLayer(nn.Module):
    """Apply self-attention and bidirectional cross-attention to both streams."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.gene_norm1 = nn.LayerNorm(d_model)
        self.gene_self_attn = _MultiheadAttention(d_model, n_heads, dropout)
        self.gene_norm2 = nn.LayerNorm(d_model)
        self.gene_ffn1 = _FeedForward(d_model, dropout)

        self.cfo_norm1 = nn.LayerNorm(d_model)
        self.cfo_self_attn = _MultiheadAttention(d_model, n_heads, dropout)
        self.cfo_norm2 = nn.LayerNorm(d_model)
        self.cfo_ffn1 = _FeedForward(d_model, dropout)

        self.gene_norm3 = nn.LayerNorm(d_model)
        self.gene_cross_attn = _MultiheadAttention(d_model, n_heads, dropout)
        self.gene_norm4 = nn.LayerNorm(d_model)
        self.gene_ffn2 = _FeedForward(d_model, dropout)

        self.cfo_norm3 = nn.LayerNorm(d_model)
        self.cfo_cross_attn = _MultiheadAttention(d_model, n_heads, dropout)
        self.cfo_norm4 = nn.LayerNorm(d_model)
        self.cfo_ffn2 = _FeedForward(d_model, dropout)

    def forward(
        self, gene_values: torch.Tensor, cfo_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_norm = self.gene_norm1(gene_values)
        gene_values = gene_values + self.gene_self_attn(gene_norm, gene_norm, gene_norm)
        gene_values = gene_values + self.gene_ffn1(self.gene_norm2(gene_values))

        cfo_norm = self.cfo_norm1(cfo_values)
        cfo_values = cfo_values + self.cfo_self_attn(cfo_norm, cfo_norm, cfo_norm)
        cfo_values = cfo_values + self.cfo_ffn1(self.cfo_norm2(cfo_values))

        gene_values = gene_values + self.gene_cross_attn(
            self.gene_norm3(gene_values),
            self.cfo_norm3(cfo_values),
            self.cfo_norm3(cfo_values),
        )
        gene_values = gene_values + self.gene_ffn2(self.gene_norm4(gene_values))

        cfo_values = cfo_values + self.cfo_cross_attn(
            self.cfo_norm3(cfo_values),
            self.gene_norm3(gene_values),
            self.gene_norm3(gene_values),
        )
        cfo_values = cfo_values + self.cfo_ffn2(self.cfo_norm4(cfo_values))
        return gene_values, cfo_values


class JointCrossTransformerEncoder(nn.Module):
    """Encode gene-only, CFO-only, or joint prepared input matrices."""

    def __init__(
        self,
        d_embedding: int,
        d_cond: int,
        gene_topk: int = 2048,
        cfo_topk: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_embedding = int(d_embedding)
        self.d_cond = int(d_cond)
        self.gene_topk = int(gene_topk)
        self.cfo_topk = int(cfo_topk)
        self.dropout = float(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_embedding))
        self.value_projection = nn.Sequential(nn.Linear(1, self.d_embedding), nn.Tanh())
        self.sign_embedding = nn.Embedding(3, self.d_embedding)
        self.layers = nn.ModuleList(
            _GeneCFOCrossLayer(self.d_embedding, n_heads, self.dropout)
            for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(self.d_embedding)
        self.output_projection = (
            nn.Identity()
            if self.d_embedding == self.d_cond
            else nn.Linear(self.d_embedding, self.d_cond)
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.sign_embedding.weight, mean=0.0, std=0.02)

    @staticmethod
    def _sign_indices(values: torch.Tensor) -> torch.Tensor:
        return torch.where(values < 0, 0, torch.where(values > 0, 2, 1)).long()

    @staticmethod
    def _select_topk(values: torch.Tensor, topk: int) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("Input values must have shape [batch, features]")
        if topk <= 0:
            raise ValueError("topk must be positive")
        return torch.topk(
            torch.abs(values), k=min(int(topk), values.shape[1]), dim=1
        ).indices

    def _tokens(
        self, values: torch.Tensor, embeddings: torch.Tensor, topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or embeddings.ndim != 2:
            raise ValueError("Input values and embeddings must both be two-dimensional")
        if values.shape[1] != embeddings.shape[0]:
            raise ValueError("Input width must match the corresponding embedding table")
        indices = self._select_topk(values, topk)
        selected = torch.gather(values, 1, indices)
        base = embeddings[indices]
        magnitude = self.value_projection(torch.log1p(torch.abs(selected)).unsqueeze(-1))
        signs = self.sign_embedding(self._sign_indices(selected))
        tokens = F.dropout(base + magnitude + signs, p=self.dropout, training=self.training)
        return tokens, indices, selected

    def forward(
        self,
        gene_values: Optional[torch.Tensor],
        h_gene: Optional[torch.Tensor],
        cfo_values: Optional[torch.Tensor],
        h_cfo: Optional[torch.Tensor],
        *,
        mode: str = "joint",
        return_explain: bool = False,
        return_token_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if mode not in {"gene_only", "cfo_only", "joint"}:
            raise ValueError(f"Unknown encode mode: {mode}")

        gene_tokens = gene_indices = gene_selected = None
        cfo_tokens = cfo_indices = cfo_selected = None
        if mode in {"gene_only", "joint"}:
            if gene_values is None or h_gene is None:
                raise ValueError("gene_values and h_gene are required for this mode")
            gene_tokens, gene_indices, gene_selected = self._tokens(
                gene_values, h_gene, self.gene_topk
            )
        if mode in {"cfo_only", "joint"}:
            if cfo_values is None or h_cfo is None:
                raise ValueError("cfo_values and h_cfo are required for this mode")
            cfo_tokens, cfo_indices, cfo_selected = self._tokens(
                cfo_values, h_cfo, self.cfo_topk
            )

        batch_size = gene_tokens.shape[0] if gene_tokens is not None else cfo_tokens.shape[0]
        cls = self.cls_token.expand(batch_size, 1, self.d_embedding)
        gene_sequence = cfo_sequence = None
        if mode == "gene_only":
            gene_sequence = torch.cat([cls, gene_tokens], dim=1)
            for layer in self.layers:
                gene_sequence, _ = layer(gene_sequence, cls)
            condition = self.output_projection(self.output_norm(gene_sequence[:, 0]))
        elif mode == "cfo_only":
            cfo_sequence = torch.cat([cls, cfo_tokens], dim=1)
            for layer in self.layers:
                _, cfo_sequence = layer(cls, cfo_sequence)
            condition = self.output_projection(self.output_norm(cfo_sequence[:, 0]))
        else:
            gene_sequence = torch.cat([cls, gene_tokens], dim=1)
            cfo_sequence = torch.cat([cls, cfo_tokens], dim=1)
            for layer in self.layers:
                gene_sequence, cfo_sequence = layer(gene_sequence, cfo_sequence)
            pooled = 0.5 * (
                self.output_norm(gene_sequence[:, 0])
                + self.output_norm(cfo_sequence[:, 0])
            )
            condition = self.output_projection(pooled)

        if not return_explain:
            return condition
        auxiliary: Dict[str, torch.Tensor] = {}
        if gene_indices is not None:
            auxiliary["gene_topk_idx"] = gene_indices
            auxiliary["gene_topk_values"] = gene_selected
        if cfo_indices is not None:
            auxiliary["cfo_topk_idx"] = cfo_indices
            auxiliary["cfo_topk_values"] = cfo_selected
        if return_token_embeddings:
            if gene_sequence is not None:
                auxiliary["gene_token_embeddings"] = gene_sequence[:, 1:, :].contiguous()
            if cfo_sequence is not None:
                auxiliary["cfo_token_embeddings"] = cfo_sequence[:, 1:, :].contiguous()
        return condition, auxiliary
