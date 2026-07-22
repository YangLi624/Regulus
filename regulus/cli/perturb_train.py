"""Implementation of ``regulus perturb-train``."""

from __future__ import annotations

import logging

import torch

from regulus.perturb.train import PerturbTrainer
from regulus.utils.reproducibility import set_random_seeds

logger = logging.getLogger(__name__)


def run_perturb_train(args) -> int:
    set_random_seeds(args.seed)
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        logger.info("Using GPU %d", args.gpu)
    else:
        logger.info("Using CPU")
    PerturbTrainer(args.config).train()
    logger.info("Training completed")
    return 0
