"""Command-line interface for Regulus."""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regulus",
        description="Perturbation driver inference, explanation, and virtual CFO manipulation",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="List or download pretrained bundles")
    download.add_argument("bundle_id", nargs="?", default=None)
    download.add_argument("--output-dir", default="bundles")
    download.add_argument("--force", action="store_true", help="Replace an installed bundle")

    bundle_build = sub.add_parser("bundle-build", help="Build a self-contained model bundle")
    bundle_build.add_argument("--bundle-id", required=True)
    bundle_build.add_argument("--config", required=True)
    bundle_build.add_argument("--perturb-checkpoint", required=True)
    bundle_build.add_argument("--graph-checkpoint", required=True)
    bundle_build.add_argument("--graph-asset-dir", required=True)
    bundle_build.add_argument("--output-dir", default="dist/bundles")

    preprocess = sub.add_parser("preprocess", help="Compute a CFO activity matrix")
    preprocess.add_argument("-i", "--input", required=True, help="Input H5AD")
    preprocess.add_argument("-o", "--output", default=None, help="Output H5AD")
    source = preprocess.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-path")
    source.add_argument("--graph-asset-dir")
    source.add_argument("--cfo-gene-sets", help="CSV with cfo_id and genes columns")
    preprocess.add_argument("--layer", default=None)
    preprocess.add_argument("--overwrite", action="store_true")

    predict = sub.add_parser("predict", help="Predict perturbation drivers")
    predict.add_argument("-i", "--input", required=True, help="Prepared H5AD")
    predict.add_argument("--bundle-path", required=True)
    predict.add_argument("--input-representation", choices=["delta", "post_state"], default=None)
    predict.add_argument("--mode", choices=["gene_only", "cfo_only", "joint"], default=None)
    predict.add_argument("--top-k", type=int, default=50)
    predict.add_argument("-o", "--output", default="predictions.csv")
    predict.add_argument("--device", default=None)

    explain = sub.add_parser("explain", help="Attribute predictions and trace graph context")
    explain.add_argument("-i", "--input", required=True, help="Prepared H5AD")
    explain.add_argument("--bundle-path", required=True)
    explain.add_argument("-o", "--output-dir", default="explain_out")
    explain.add_argument("--mode", choices=["gene_only", "cfo_only", "joint"], default=None)
    explain.add_argument(
        "--attribution-channel", choices=["gene", "cfo", "both"], default=None
    )
    explain.add_argument(
        "--attribution-method",
        choices=["gradient_x_input", "integrated_gradients"],
        default="gradient_x_input",
    )
    explain.add_argument("--target", default=None, help="Candidate name; default is top prediction")
    explain.add_argument("--ig-steps", type=int, default=16)
    explain.add_argument("--top-k-features", type=int, default=20)
    explain.add_argument("--top-k-candidates", type=int, default=1)
    explain.add_argument("--obs-filter", default=None, help="Pandas query applied to cells")
    explain.add_argument("--plot", action="store_true", help="Render an optional network PDF")
    explain.add_argument("--device", default=None)

    manipulate = sub.add_parser("manipulate", help="Run a virtual CFO edit")
    manipulate.add_argument("--bundle-path", required=True)
    manipulate.add_argument("--anchor-h5ad", required=True)
    manipulate.add_argument("--cfo-targets", required=True, help="Comma-separated CFO IDs or names")
    manipulate.add_argument("--cfo-delta", default="0.2", help="One or more comma-separated values")
    manipulate.add_argument(
        "--edit-mode", default="anchor_plus_delta", choices=["anchor_plus_delta"]
    )
    manipulate.add_argument("--mode", default="cfo_only", choices=["cfo_only", "joint"])
    manipulate.add_argument("-o", "--output-dir", default="manipulate_out")
    manipulate.add_argument("--obs-filter", default=None)
    manipulate.add_argument("--top-k", type=int, default=20)
    manipulate.add_argument(
        "--sort-by", choices=["rank_gain", "delta", "after"], default="rank_gain"
    )
    manipulate.add_argument("--device", default=None)

    perturb_train = sub.add_parser("perturb-train", help="Train a perturbation model")
    perturb_train.add_argument("--config", required=True)
    perturb_train.add_argument("--seed", type=int, default=42)
    perturb_train.add_argument("--gpu", type=int, default=0)

    graph_train = sub.add_parser("graph-train", help="Train the graph model")
    graph_train.add_argument("--config", default="configs/graph_config.yaml")
    graph_train.add_argument(
        "--graph-asset-dir", "--data-dir", dest="graph_asset_dir", default=None
    )
    graph_train.add_argument("--output-dir", default="outputs")
    graph_train.add_argument("--output-suffix", default="graph")
    graph_train.add_argument("--resume", default=None)
    graph_train.add_argument("--seed", type=int, default=42)
    graph_train.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    handlers = {
        "download": ("regulus.cli.download", "run_download"),
        "bundle-build": ("regulus.cli.bundle_build", "run_bundle_build"),
        "preprocess": ("regulus.cli.preprocess", "run_preprocess"),
        "predict": ("regulus.cli.predict", "run_predict"),
        "explain": ("regulus.cli.explain", "run_explain"),
        "manipulate": ("regulus.cli.manipulate", "run_manipulate"),
        "perturb-train": ("regulus.cli.perturb_train", "run_perturb_train"),
        "graph-train": ("regulus.cli.graph_train", "run_graph_train"),
    }
    module_name, function_name = handlers[args.command]
    module = __import__(module_name, fromlist=[function_name])
    return int(getattr(module, function_name)(args))


if __name__ == "__main__":
    sys.exit(main())
