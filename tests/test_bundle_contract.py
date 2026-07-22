"""Portable bundle path and schema tests."""

import json
from pathlib import Path

import pytest

from regulus.io.bundle import load_bundle_manifest
from regulus.io.package import _portable_config


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.yaml").write_text(
        "model:\n  representation: post_state\n  channels: gene\n  head: mlp\n",
        encoding="utf-8",
    )
    for name in ("perturb.pt", "graph.pt", "genes.csv"):
        (bundle / name).write_bytes(b"fixture")
    (bundle / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0",
        "bundle_id": "fixture",
        "representation": "post_state",
        "channels": "gene",
        "head": "mlp",
        "train_config": "config.yaml",
        "perturb_ckpt": "perturb.pt",
        "graph_ckpt": "graph.pt",
        "gene_universe": "genes.csv",
    }), encoding="utf-8")
    return bundle


def test_bundle_paths_are_relative_to_bundle_root(tmp_path: Path):
    manifest = load_bundle_manifest(_bundle(tmp_path))
    assert manifest.bundle_id == "fixture"
    assert manifest.train_config.parent == manifest.bundle_dir


def test_bundle_rejects_parent_path_escape(tmp_path: Path):
    bundle = _bundle(tmp_path)
    raw = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    raw["graph_ckpt"] = "../outside.pt"
    (bundle / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_bundle_manifest(bundle)


def test_packaged_training_config_uses_portable_paths():
    config = {
        "data": {
            "dataset_dir": "tasks/private/data",
            "graph_asset_dir": "assets/private/graph",
            "gene_universe": "assets/private/genes.csv",
        },
        "model": {"graph_checkpoint": "tasks/private/graph.pt"},
        "output_dir": "tasks/private/output",
        "checkpoint_dir": "tasks/private/checkpoints",
        "resume": True,
    }
    portable = _portable_config(config)
    assert portable["data"] == {
        "dataset_dir": "data",
        "graph_asset_dir": "graph",
        "gene_universe": "graph/universes/gene_universe.csv",
    }
    assert portable["model"]["graph_checkpoint"] == "checkpoints/graph_model.pt"
    assert portable["output_dir"] == "outputs"
    assert portable["checkpoint_dir"] == "outputs/checkpoints"
    assert portable["resume"] is False
    assert config["data"]["dataset_dir"] == "tasks/private/data"
