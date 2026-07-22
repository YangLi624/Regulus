"""Feature attribution for Regulus perturbation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from regulus.perturb.model import RegulusPerturbationModel
from regulus.perturb.spec import normalize_mode

ATTRIBUTION_METHODS = ("gradient_x_input", "integrated_gradients")
ATTRIBUTION_CHANNELS = ("gene", "cfo", "both")


@dataclass
class AttributionBatch:
    logits: torch.Tensor
    target_indices: torch.Tensor
    token_indices: dict[str, torch.Tensor]
    input_values: dict[str, torch.Tensor]
    signed_attributions: dict[str, torch.Tensor]


def normalize_attribution_method(value: str) -> str:
    method = str(value)
    if method not in ATTRIBUTION_METHODS:
        raise ValueError(f"attribution_method must be one of {ATTRIBUTION_METHODS}")
    return method


def normalize_attribution_channel(value: str) -> str:
    channel = str(value)
    if channel not in ATTRIBUTION_CHANNELS:
        raise ValueError(f"attribution_channel must be one of {ATTRIBUTION_CHANNELS}")
    return channel


def _available_channels(mode: str) -> set[str]:
    return {
        "gene_only": {"gene"},
        "cfo_only": {"cfo"},
        "joint": {"gene", "cfo"},
    }[mode]


def _selected_channels(channel: str) -> tuple[str, ...]:
    return ("gene", "cfo") if channel == "both" else (channel,)


def _forward_with_aux(
    model: RegulusPerturbationModel,
    gene_input: Optional[torch.Tensor],
    cfo_input: Optional[torch.Tensor],
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    condition, aux = model.encode(
        gene_input=gene_input,
        cfo_input=cfo_input,
        mode=mode,
        return_explain=True,
    )
    return model.head(condition), aux


def _target_indices(logits: torch.Tensor, requested: Optional[torch.Tensor]) -> torch.Tensor:
    target = logits.argmax(dim=1) if requested is None else requested.to(logits.device)
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError("target_indices must have shape [batch]")
    if torch.any(target < 0) or torch.any(target >= logits.shape[1]):
        raise IndexError("target_indices contains an invalid candidate index")
    return target.to(torch.long)


def _token_indices(aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if "topk_idx" in aux:
        result["gene"] = aux["topk_idx"].detach()
    if "gene_topk_idx" in aux:
        result["gene"] = aux["gene_topk_idx"].detach()
    if "cfo_topk_idx" in aux:
        result["cfo"] = aux["cfo_topk_idx"].detach()
    return result


def attribute_batch(
    model: RegulusPerturbationModel,
    *,
    gene_input: Optional[torch.Tensor],
    cfo_input: Optional[torch.Tensor],
    mode: str,
    attribution_channel: str = "gene",
    attribution_method: str = "gradient_x_input",
    target_indices: Optional[torch.Tensor] = None,
    ig_steps: int = 16,
) -> AttributionBatch:
    """Attribute a fixed candidate score to one or both model input channels."""
    mode = normalize_mode(mode, model.channels)
    channel = normalize_attribution_channel(attribution_channel)
    method = normalize_attribution_method(attribution_method)
    selected_channels = _selected_channels(channel)
    unavailable = set(selected_channels).difference(_available_channels(mode))
    if unavailable:
        raise ValueError(
            f"Cannot attribute {sorted(unavailable)} while runtime mode is {mode!r}"
        )
    inputs = {"gene": gene_input, "cfo": cfo_input}
    for selected in selected_channels:
        if inputs[selected] is None:
            raise ValueError(f"{selected}_input is required for {channel} attribution")

    with torch.no_grad():
        original_logits, original_aux = _forward_with_aux(model, gene_input, cfo_input, mode)
    target = _target_indices(original_logits, target_indices)

    if method == "gradient_x_input":
        differentiable: dict[str, torch.Tensor] = {}
        forward_inputs: dict[str, Optional[torch.Tensor]] = {}
        for name, value in inputs.items():
            if value is None:
                forward_inputs[name] = None
            elif name in selected_channels:
                differentiable[name] = value.detach().clone().requires_grad_(True)
                forward_inputs[name] = differentiable[name]
            else:
                forward_inputs[name] = value.detach()
        logits, _ = _forward_with_aux(
            model, forward_inputs["gene"], forward_inputs["cfo"], mode
        )
        selected_score = logits.gather(1, target[:, None]).sum()
        gradients = torch.autograd.grad(
            selected_score,
            [differentiable[name] for name in selected_channels],
            retain_graph=False,
        )
        attributions = {
            name: differentiable[name].detach() * gradient.detach()
            for name, gradient in zip(selected_channels, gradients)
        }
    else:
        if ig_steps < 1:
            raise ValueError("ig_steps must be positive")
        totals = {
            name: torch.zeros_like(inputs[name])  # type: ignore[arg-type]
            for name in selected_channels
        }
        for step in range(1, ig_steps + 1):
            alpha = float(step) / float(ig_steps)
            differentiable = {}
            forward_inputs = {}
            for name, value in inputs.items():
                if value is None:
                    forward_inputs[name] = None
                elif name in selected_channels:
                    differentiable[name] = (value.detach() * alpha).requires_grad_(True)
                    forward_inputs[name] = differentiable[name]
                else:
                    forward_inputs[name] = value.detach()
            logits, _ = _forward_with_aux(
                model, forward_inputs["gene"], forward_inputs["cfo"], mode
            )
            selected_score = logits.gather(1, target[:, None]).sum()
            gradients = torch.autograd.grad(
                selected_score,
                [differentiable[name] for name in selected_channels],
                retain_graph=False,
            )
            for name, gradient in zip(selected_channels, gradients):
                totals[name] = totals[name] + gradient.detach()
        attributions = {
            name: inputs[name].detach() * totals[name] / float(ig_steps)  # type: ignore[union-attr]
            for name in selected_channels
        }

    return AttributionBatch(
        logits=original_logits.detach(),
        target_indices=target.detach(),
        token_indices=_token_indices(original_aux),
        input_values={
            name: inputs[name].detach()  # type: ignore[union-attr]
            for name in selected_channels
        },
        signed_attributions=attributions,
    )
