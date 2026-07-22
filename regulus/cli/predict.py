"""Implementation of ``regulus predict``."""

from __future__ import annotations

import logging
from pathlib import Path

from regulus.perturb.inference import RegulusModel

logger = logging.getLogger(__name__)


def run_predict(args) -> int:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input H5AD not found: {input_path}")
    model = RegulusModel.from_bundle(
        args.bundle_path,
        input_representation=args.input_representation,
        device=args.device,
    )
    frame = model.predict(input_path, mode=args.mode, top_k=args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.info("Wrote %d predictions to %s", len(frame), output)
    return 0
