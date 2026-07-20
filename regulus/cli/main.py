"""Command-line entry point for the Regulus graph release."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regulus",
        description="Regulus heterogeneous graph training",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    graph = subparsers.add_parser("graph-train", help="Train the HGT graph model")
    graph.add_argument("--config", default="configs/graph_config.yaml")
    graph.add_argument(
        "--graph-asset-dir",
        type=str,
        default=None,
        help="Versioned graph asset root; overrides data.graph_asset_dir",
    )
    graph.add_argument("--output-dir", default="outputs")
    graph.add_argument("--output-suffix", default="graph")
    graph.add_argument("--resume", default=None)
    graph.add_argument("--seed", type=int, default=42)
    graph.add_argument("--gpu", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "graph-train":
        from regulus.cli.graph_train import run_graph_train

        return run_graph_train(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
