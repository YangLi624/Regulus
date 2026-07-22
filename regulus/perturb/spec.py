"""Canonical perturbation-model configuration.

Model construction (``channels``) is separate from runtime ablation
(``mode``). A gene-only pass through a dual-channel checkpoint therefore
keeps that checkpoint's joint encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


REPRESENTATIONS = ("delta", "post_state")
CHANNELS = ("gene", "cfo", "gene_cfo")
HEADS = ("mlp", "prototype_matching")
CHANNEL_MODES = ("joint", "gene_only", "cfo_only")

def normalize_representation(value: str) -> str:
    value = str(value)
    if value not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {value!r}")
    return value


def normalize_channels(value: str) -> str:
    value = str(value)
    if value not in CHANNELS:
        raise ValueError(f"channels must be one of {CHANNELS}, got {value!r}")
    return value


def normalize_head(value: str) -> str:
    value = str(value)
    if value not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {value!r}")
    return value


def normalize_mode(value: Optional[str], channels: str) -> str:
    channels = normalize_channels(channels)
    if value is None:
        return {"gene": "gene_only", "cfo": "cfo_only", "gene_cfo": "joint"}[channels]
    mode = str(value)
    if mode not in CHANNEL_MODES:
        raise ValueError(f"mode must be one of {CHANNEL_MODES}, got {value!r}")
    allowed = {
        "gene": {"gene_only"},
        "cfo": {"cfo_only"},
        "gene_cfo": set(CHANNEL_MODES),
    }[channels]
    if mode not in allowed:
        raise ValueError(f"mode={mode!r} is not available for channels={channels!r}")
    return mode


@dataclass(frozen=True)
class PerturbModelSpec:
    representation: str
    channels: str
    head: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PerturbModelSpec":
        model = config.get("model", config)
        return cls(
            representation=normalize_representation(str(model.get("representation", "post_state"))),
            channels=normalize_channels(str(model.get("channels", "gene_cfo"))),
            head=normalize_head(str(model.get("head", "prototype_matching"))),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "representation": self.representation,
            "channels": self.channels,
            "head": self.head,
        }
