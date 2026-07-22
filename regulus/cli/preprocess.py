"""Implementation of ``regulus preprocess``."""

from __future__ import annotations

import logging
from pathlib import Path

from regulus.io.bundle import load_bundle_manifest
from regulus.preprocess.ucell import preprocess_h5ad_ucell

logger = logging.getLogger(__name__)


def run_preprocess(args) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.regulus.h5ad")
    )
    graph_asset_dir = getattr(args, "graph_asset_dir", None)
    if getattr(args, "bundle_path", None):
        manifest = load_bundle_manifest(args.bundle_path)
        if manifest.graph_asset_dir is None:
            raise ValueError("The selected bundle does not declare a graph asset")
        graph_asset_dir = manifest.graph_asset_dir

    preprocess_h5ad_ucell(
        input_path,
        output_path,
        cfo_gene_sets=getattr(args, "cfo_gene_sets", None),
        graph_asset_dir=graph_asset_dir,
        layer=getattr(args, "layer", None),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    logger.info("Preprocessing complete: %s", output_path)
    return 0
