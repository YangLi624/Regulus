"""Implementation of ``regulus manipulate``."""

from __future__ import annotations

import logging
from pathlib import Path

from regulus.perturb.inference import RegulusModel

logger = logging.getLogger(__name__)


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def run_manipulate(args) -> int:
    model = RegulusModel.from_bundle(args.bundle_path, device=args.device)
    frame = model.manipulate(
        cfo_targets=_parse_strings(args.cfo_targets),
        cfo_delta=_parse_floats(args.cfo_delta),
        anchor_h5ad=args.anchor_h5ad,
        obs_filter=args.obs_filter,
        mode=args.mode,
        edit_mode=args.edit_mode,
        top_k=args.top_k,
        sort_by=args.sort_by,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "manipulation_rankings.csv"
    frame.to_csv(output, index=False)
    logger.info("Wrote %d manipulation rows to %s", len(frame), output)
    return 0
