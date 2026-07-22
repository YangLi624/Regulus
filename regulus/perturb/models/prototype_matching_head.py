"""Prototype-matching classifier for TF and gene perturbations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMatchingHead(nn.Module):
    """Match a condition embedding to frozen HGT candidate prototypes."""

    def __init__(
        self,
        d_cond: int,
        h_prototypes: torch.Tensor,
        use_bilinear: bool = True,
        learnable_temperature: bool = True,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        if h_prototypes.ndim != 2:
            raise ValueError("h_prototypes must have shape [n_candidates, d_embedding]")
        self.d_cond = int(d_cond)
        self.d_embedding = int(h_prototypes.shape[1])
        self.use_bilinear = bool(use_bilinear)
        self.cond_proj = (
            nn.Identity()
            if self.d_cond == self.d_embedding
            else nn.Linear(self.d_cond, self.d_embedding)
        )
        if self.use_bilinear:
            self.M = nn.Parameter(torch.eye(self.d_embedding))
        else:
            self.register_parameter("M", None)

        prototypes = h_prototypes.detach().clone()
        prototypes.requires_grad = False
        self.register_buffer("h_prototypes", prototypes)

        temperature_init = float(temperature_init)
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive")
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
            self.register_buffer("temperature", torch.tensor(1.0))
        else:
            self.log_temperature = None
            self.register_buffer("temperature", torch.tensor(temperature_init))

    @property
    def n_candidates(self) -> int:
        return int(self.h_prototypes.shape[0])

    def _get_temperature(self) -> torch.Tensor:
        if self.log_temperature is not None:
            return F.softplus(self.log_temperature)
        return self.temperature

    def forward(self, condition_embedding: torch.Tensor) -> torch.Tensor:
        if condition_embedding.ndim != 2 or condition_embedding.shape[1] != self.d_cond:
            raise ValueError(
                f"condition_embedding must have shape [batch, {self.d_cond}], "
                f"got {tuple(condition_embedding.shape)}"
            )
        projected = self.cond_proj(condition_embedding)
        if self.M is not None:
            projected = projected @ self.M
        return (projected @ self.h_prototypes.t()) / self._get_temperature()
