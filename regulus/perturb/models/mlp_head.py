"""Compact MLP perturbation classifier."""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        d_cond: int,
        n_candidates: int,
        hidden_multiplier: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_cond = int(d_cond)
        self.n_candidates = int(n_candidates)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_cond),
            nn.Dropout(dropout),
            nn.Linear(self.d_cond, self.d_cond * hidden_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_cond * hidden_multiplier, self.n_candidates),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, condition_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(condition_embedding)
