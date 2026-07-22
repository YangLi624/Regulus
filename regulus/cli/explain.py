"""Implementation of ``regulus explain``."""

from __future__ import annotations

import logging
from pathlib import Path

from regulus.perturb.inference import RegulusModel

logger = logging.getLogger(__name__)


def run_explain(args) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input H5AD not found: {input_path}")
    model = RegulusModel.from_bundle(args.bundle_path, device=args.device)
    outputs = model.explain(
        input_path,
        mode=args.mode,
        output_dir=args.output_dir,
        attribution_channel=args.attribution_channel,
        attribution_method=args.attribution_method,
        target_candidate=args.target,
        ig_steps=args.ig_steps,
        top_k_features=args.top_k_features,
        top_k_candidates=args.top_k_candidates,
        obs_filter=args.obs_filter,
        render_pdf=args.plot,
    )
    logger.info("Scores: %s", outputs.scores_csv)
    logger.info("Attributions: %s", outputs.attributions_csv)
    logger.info("Evidence paths: %s", outputs.evidence_jsonl)
    logger.info("Manifest: %s", outputs.manifest_json)
    return 0
