from regulus.cli.main import build_parser


def test_graph_train_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "graph-train",
            "--config",
            "configs/graph_config.yaml",
            "--graph-asset-dir",
            "assets/graph/regulus_graph_v1",
        ]
    )
    assert args.command == "graph-train"
    assert args.graph_asset_dir == "assets/graph/regulus_graph_v1"
