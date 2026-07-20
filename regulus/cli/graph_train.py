"""CLI adapter for Regulus HGT graph pretraining."""

from __future__ import annotations

import logging
import os

from regulus.graph.train import run_graph_training_from_args

logger = logging.getLogger(__name__)


def run_graph_train(args) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    final_path = run_graph_training_from_args(args)
    logger.info("graph-train finished: %s", final_path)
    return 0
