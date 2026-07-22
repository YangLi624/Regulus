"""Encode a preprocessed gene matrix with frozen HGT gene embeddings.

For each sample, the encoder selects genes by absolute input magnitude,
combines their frozen graph embeddings with value and sign features, and
aggregates the sequence through a small Transformer and a CLS token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphBiasConfig:
    """Optional additive attention bias derived from graph structure."""

    use_graph_bias: bool = False
    graph_bias_type: str = "none"
    graph_bias_strength: float = 1.0


class _MultiheadSelfAttentionWithBias(nn.Module):
    """Batch-first multi-head self-attention with an additive bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
            attn_bias:
                - None
                - [B, 1, L, L] or [B, H, L, L], added to QK^T
        Returns:
            y: [B, L, d_model]
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3D [B, L, d_model], got {x.shape}")
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, L, d_head]
        q = q.view(bsz, seqlen, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.d_head).transpose(1, 2)

        # Use PyTorch SDPA (Flash / mem-efficient attention when available).
        # This avoids materializing the full [B,H,L,L] attention matrix in memory.
        if attn_bias is not None:
            if attn_bias.ndim != 4:
                raise ValueError(f"attn_bias must be 4D, got {attn_bias.shape}")
            if attn_bias.shape[0] != bsz or attn_bias.shape[-1] != seqlen or attn_bias.shape[-2] != seqlen:
                raise ValueError(
                    f"attn_bias shape mismatch: bias={attn_bias.shape}, expected [B,*,L,L] with B={bsz}, L={seqlen}"
                )
            attn_mask = attn_bias.to(dtype=q.dtype)
        else:
            attn_mask = None

        dropout_p = self.dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )  # [B, H, L, d_head]
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)  # [B, L, d_model]
        y = self.dropout(self.out(y))
        return y


class _TransformerEncoderLayerWithBias(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiheadSelfAttentionWithBias(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class GeneTokenTransformerEncoder(nn.Module):
    """Map gene inputs and frozen gene embeddings to condition embeddings."""

    def __init__(
        self,
        d_embedding: int,
        d_cond: int,
        topk: int = 2048,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        graph_bias: Optional[GraphBiasConfig] = None,
    ) -> None:
        super().__init__()
        self.d_embedding = int(d_embedding)
        self.d_cond = int(d_cond)
        self.topk = int(topk)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.graph_bias = graph_bias or GraphBiasConfig()

        # Project input magnitudes to the frozen embedding dimension.
        self.value_projection = nn.Sequential(
            nn.Linear(1, self.d_embedding),
            nn.Tanh(),
        )
        # sign embedding: -1,0,+1 -> 3
        self.sign_emb = nn.Embedding(3, self.d_embedding)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_embedding))
        self.enc_layers = nn.ModuleList(
            [
                _TransformerEncoderLayerWithBias(
                    d_model=self.d_embedding,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    ffn_mult=4,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_embedding)
        self.out_proj = nn.Identity() if self.d_embedding == self.d_cond else nn.Linear(self.d_embedding, self.d_cond)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # sign_emb init
        nn.init.normal_(self.sign_emb.weight, mean=0.0, std=0.02)

    @staticmethod
    def _sign_to_index(values: torch.Tensor) -> torch.Tensor:
        """Map values to indices for negative, zero, and positive signs."""
        neg = (values < 0).to(torch.long)
        pos = (values > 0).to(torch.long)
        # neg->0, zero->1, pos->2
        return (pos * 2 + (1 - neg - pos) * 1 + neg * 0).to(torch.long)

    def _select_topk(self, gene_values: torch.Tensor) -> torch.Tensor:
        """Return top-k gene indices with shape [B, K]."""
        if gene_values.ndim != 2:
            raise ValueError(f"gene_values must be 2D [batch, genes], got {gene_values.shape}")
        bsz, n_genes = gene_values.shape
        k = min(self.topk, n_genes)
        magnitudes = torch.abs(gene_values)
        _, idx = torch.topk(magnitudes, k=k, dim=1, largest=True, sorted=True)
        return idx

    def forward(
        self,
        gene_values: torch.Tensor,
        h_gene: torch.Tensor,  # [n_genes, d_embedding] (frozen)
        graph_bias_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_explain: bool = False,
        return_token_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if h_gene.ndim != 2 or h_gene.shape[1] != self.d_embedding:
            raise ValueError(f"h_gene must be [n_genes, d_embedding={self.d_embedding}], got {h_gene.shape}")
        if gene_values.shape[1] != h_gene.shape[0]:
            raise ValueError(
                f"gene input width ({gene_values.shape[1]}) does not match "
                f"the gene embedding table ({h_gene.shape[0]})"
            )

        idx = self._select_topk(gene_values)
        bsz, k = idx.shape

        # Token content combines frozen graph context with value features.
        gene_tok = h_gene[idx]  # [B, K, d_embedding]
        selected_values = torch.gather(gene_values, dim=1, index=idx)

        # delta magnitude feature
        magnitude = torch.log1p(torch.abs(selected_values)).unsqueeze(-1)
        magnitude_embedding = self.value_projection(magnitude)
        sign_idx = self._sign_to_index(selected_values)
        sign_emb = self.sign_emb(sign_idx)  # [B, K, d_embedding]

        x = gene_tok + magnitude_embedding + sign_emb
        x = F.dropout(x, p=self.dropout, training=self.training)

        # prepend CLS
        cls = self.cls_token.expand(bsz, 1, self.d_embedding)
        x = torch.cat([cls, x], dim=1)  # [B, L=K+1, d_embedding]
        seqlen = x.shape[1]

        attn_bias = None
        if self.graph_bias.use_graph_bias:
            if graph_bias_fn is None:
                raise RuntimeError("use_graph_bias=True but graph_bias_fn is None")
            # bias for tokens only (KxK); we pad to include CLS at position 0 with 0 bias.
            token_bias = graph_bias_fn(idx)  # expected [B, 1, K, K] or [B, H, K, K]
            if token_bias.shape[-1] != k or token_bias.shape[-2] != k or token_bias.shape[0] != bsz:
                raise ValueError(f"graph_bias_fn returned {token_bias.shape}, expected [B,*,K,K] with K={k}")
            # create [B, heads_or_1, L, L]
            heads_dim = token_bias.shape[1]
            attn_bias = torch.zeros((bsz, heads_dim, seqlen, seqlen), device=token_bias.device, dtype=token_bias.dtype)
            attn_bias[:, :, 1:, 1:] = token_bias

        # encode
        for layer in self.enc_layers:
            x = layer(x, attn_bias=attn_bias)

        cls_out = self.out_norm(x[:, 0])  # [B, d_embedding]
        cond = self.out_proj(cls_out)  # [B, d_cond]

        if not return_explain:
            return cond

        aux: Dict[str, torch.Tensor] = {
            "topk_idx": idx,
            "topk_values": selected_values,
        }
        if return_token_embeddings:
            # tokens before CLS and before encoder, useful for token-level attribution.
            aux["token_emb"] = x[:, 1:, :].contiguous()
        return cond, aux
