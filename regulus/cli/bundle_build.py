"""Implementation of ``regulus bundle-build``."""

from __future__ import annotations

from regulus.io.package import build_release_bundle


def run_bundle_build(args) -> int:
    archive = build_release_bundle(
        bundle_id=args.bundle_id,
        config_path=args.config,
        perturb_checkpoint=args.perturb_checkpoint,
        graph_checkpoint=args.graph_checkpoint,
        graph_asset_dir=args.graph_asset_dir,
        output_dir=args.output_dir,
    )
    print(archive)
    return 0
