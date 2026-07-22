"""Stable file outputs for Regulus attribution and graph traversal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from regulus.explain.attribution import attribute_batch
from regulus.explain.evidence import EvidenceGraphBuilder
from regulus.explain.viz import render_evidence_network_pdfs
from regulus.io.bundle import sha256_file

EXPLAIN_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ExplainOutputs:
    output_dir: Path
    scores_csv: Path
    attributions_csv: Path
    evidence_jsonl: Path
    manifest_json: Path
    pdf_paths: tuple[Path, ...] = ()

def _collate(samples: list[dict]) -> dict[str, Any]:
    batch: dict[str, Any] = {"cell_id": [str(sample.get("cell_id", "")) for sample in samples]}
    for key in ("gene_input", "cfo_input"):
        values = [sample.get(key) for sample in samples]
        if values and all(value is not None for value in values):
            batch[key] = torch.stack([
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                for value in values
            ])
    return batch


def _candidate_order_hash(candidates: list[str]) -> str:
    payload = "\n".join(candidates).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _feature_name(
    channel: str,
    index: int,
    gene_symbols: list[str],
    evidence_builder: EvidenceGraphBuilder,
) -> str:
    if channel == "gene":
        return gene_symbols[index] if 0 <= index < len(gene_symbols) else f"gene_{index}"
    return evidence_builder.cfo_idx_to_id.get(index, f"CFO_{index}")


def run_explain(
    regulus_model,
    dataset: Dataset,
    output_dir: Union[str, Path],
    *,
    mode: str,
    attribution_channel: str = "gene",
    attribution_method: str = "gradient_x_input",
    target_candidate: Optional[str] = None,
    ig_steps: int = 16,
    top_k_features: int = 20,
    top_k_candidates: int = 1,
    render_pdf: bool = False,
    pdf_cell_id: Optional[str] = None,
    batch_size: int = 8,
) -> ExplainOutputs:
    """Write scores, long-form attributions, evidence paths, and a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = regulus_model.model
    model.eval()
    candidates = list(regulus_model.candidate_perturbations)
    target_index = None
    if target_candidate is not None:
        if target_candidate not in candidates:
            raise KeyError(f"Unknown target candidate: {target_candidate}")
        target_index = candidates.index(target_candidate)

    universe = pd.read_csv(regulus_model.manifest.gene_universe)
    gene_symbols = list(universe["gene_symbol"].astype(str))
    evidence_builder = EvidenceGraphBuilder(
        regulus_model._hgt,
        gene_symbols=gene_symbols,
        tf_candidate_names=candidates,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    all_cell_ids: list[str] = []
    all_scores: list[np.ndarray] = []
    attribution_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    for batch in loader:
        cell_ids = list(batch["cell_id"])
        gene_input = batch.get("gene_input")
        cfo_input = batch.get("cfo_input")
        if gene_input is not None:
            gene_input = gene_input.to(regulus_model.device)
        if cfo_input is not None:
            cfo_input = cfo_input.to(regulus_model.device)
        requested_target = None
        if target_index is not None:
            requested_target = torch.full(
                (len(cell_ids),), target_index, dtype=torch.long, device=regulus_model.device
            )
        attributed = attribute_batch(
            model,
            gene_input=gene_input,
            cfo_input=cfo_input,
            mode=mode,
            attribution_channel=attribution_channel,
            attribution_method=attribution_method,
            target_indices=requested_target,
            ig_steps=ig_steps,
        )
        logits = attributed.logits.cpu()
        targets = attributed.target_indices.cpu()
        all_cell_ids.extend(cell_ids)
        all_scores.append(logits.numpy())

        sample_features: list[dict[str, tuple[list[int], list[float]]]] = [
            {} for _ in cell_ids
        ]
        for channel, full_attribution in attributed.signed_attributions.items():
            token_indices = attributed.token_indices[channel].cpu()
            attribution_values = full_attribution.detach().cpu()
            input_values = attributed.input_values[channel].detach().cpu()
            selected_attr = torch.gather(attribution_values, 1, token_indices)
            selected_input = torch.gather(input_values, 1, token_indices)
            for sample_index, cell_id in enumerate(cell_ids):
                count = min(int(top_k_features), token_indices.shape[1])
                order = torch.argsort(selected_attr[sample_index].abs(), descending=True)[:count]
                indices = token_indices[sample_index, order].tolist()
                signed = selected_attr[sample_index, order].tolist()
                inputs = selected_input[sample_index, order].tolist()
                sample_features[sample_index][channel] = (indices, signed)
                for rank, (feature_index, input_value, score) in enumerate(
                    zip(indices, inputs, signed), start=1
                ):
                    attribution_rows.append({
                        "cell_id": cell_id,
                        "target_candidate": candidates[int(targets[sample_index])],
                        "target_index": int(targets[sample_index]),
                        "channel": channel,
                        "feature_index": int(feature_index),
                        "feature_id": _feature_name(
                            channel, int(feature_index), gene_symbols, evidence_builder
                        ),
                        "input_value": float(input_value),
                        "signed_attribution": float(score),
                        "absolute_attribution": float(abs(score)),
                        "rank": rank,
                    })

        top_count = min(max(1, int(top_k_candidates)), logits.shape[1])
        top_scores, top_indices = torch.topk(logits, k=top_count, dim=1)
        for sample_index, cell_id in enumerate(cell_ids):
            predicted_drivers = (
                [target_candidate]
                if target_candidate is not None
                else [
                    candidates[int(index)]
                    for index in top_indices[sample_index].tolist()
                ]
            )
            paths = []
            gene_features = sample_features[sample_index].get("gene")
            if gene_features is not None:
                paths.extend(evidence_builder.trace_from_genes(
                    *gene_features,
                    candidate_names=predicted_drivers,
                ))
            cfo_features = sample_features[sample_index].get("cfo")
            if cfo_features is not None:
                paths.extend(evidence_builder.trace_from_cfos(
                    *cfo_features,
                    candidate_names=predicted_drivers,
                ))
            gene_drivers = [
                name for name in predicted_drivers
                if name not in evidence_builder.tf_symbol_to_idx
                and name in evidence_builder.gene_symbol_to_idx
            ]
            if gene_drivers:
                paths.extend(evidence_builder.trace_from_candidates(gene_drivers))
            evidence_rows.append({
                "cell_id": cell_id,
                "target_candidate": candidates[int(targets[sample_index])],
                "target_rule": "explicit" if target_candidate is not None else "top_prediction",
                "graph_drivers": predicted_drivers,
                "top_candidates": [
                    {
                        "candidate": candidates[int(index)],
                        "score": float(score),
                    }
                    for index, score in zip(
                        top_indices[sample_index].tolist(), top_scores[sample_index].tolist()
                    )
                ],
                "paths": evidence_builder.to_jsonable(paths),
            })

    scores = np.concatenate(all_scores) if all_scores else np.empty((0, len(candidates)))
    scores_csv = output_dir / "scores.csv"
    attributions_csv = output_dir / "attributions.csv"
    evidence_jsonl = output_dir / "evidence_paths.jsonl"
    manifest_json = output_dir / "explain_manifest.json"
    score_frame = pd.DataFrame(scores, index=all_cell_ids, columns=candidates)
    score_frame.index.name = "cell_id"
    score_frame.to_csv(scores_csv)
    pd.DataFrame(attribution_rows).to_csv(attributions_csv, index=False)
    with evidence_jsonl.open("w", encoding="utf-8") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": EXPLAIN_SCHEMA_VERSION,
        "bundle_id": regulus_model.manifest.bundle_id,
        "bundle_schema_version": regulus_model.manifest.schema_version,
        "perturb_checkpoint_sha256": sha256_file(regulus_model.manifest.perturb_ckpt),
        "graph_asset_id": regulus_model.manifest.graph_asset_id,
        "representation": regulus_model.input_representation,
        "trained_channels": regulus_model.channels,
        "runtime_mode": mode,
        "attribution_channel": attribution_channel,
        "attribution_method": attribution_method,
        "target_rule": "explicit" if target_candidate is not None else "top_prediction",
        "target_candidate": target_candidate,
        "ig_steps": int(ig_steps) if attribution_method == "integrated_gradients" else None,
        "top_k_features": int(top_k_features),
        "top_k_candidates": int(top_k_candidates),
        "candidate_order_sha256": _candidate_order_hash(candidates),
        "evidence_sources": ["reference_supported", "llm_context"],
        "outputs": [scores_csv.name, attributions_csv.name, evidence_jsonl.name],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    pdf_paths: tuple[Path, ...] = ()
    if render_pdf:
        pdf_paths = tuple(render_evidence_network_pdfs(
            evidence_jsonl,
            output_dir / "network_viz",
            cell_id=pdf_cell_id or (all_cell_ids[0] if all_cell_ids else None),
        ))
    return ExplainOutputs(
        output_dir=output_dir,
        scores_csv=scores_csv,
        attributions_csv=attributions_csv,
        evidence_jsonl=evidence_jsonl,
        manifest_json=manifest_json,
        pdf_paths=pdf_paths,
    )
