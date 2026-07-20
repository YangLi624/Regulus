import json
from pathlib import Path

from regulus.graph.schema import MESSAGE_PASSING_EDGE_TYPES, TF_CFO_LLM, TRAINED_EDGE_TYPES


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_graph_schema() -> None:
    assert MESSAGE_PASSING_EDGE_TYPES == (
        ("tf", "regulates", "gene"),
        ("gene", "annotated_to", "go"),
        ("celltype", "scenic_activity", "tf"),
        ("celltype", "llm_context", "go"),
        ("tf", "llm_regulates", "go"),
    )
    assert TF_CFO_LLM in MESSAGE_PASSING_EDGE_TYPES
    assert TF_CFO_LLM not in TRAINED_EDGE_TYPES


def test_public_asset_manifest_scope() -> None:
    manifest_path = ROOT / "assets/graph/regulus_graph_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in manifest["files"]]

    assert len(paths) == 19
    assert "universes/tf_universe.csv" in paths
    assert not any("provenance/" in path for path in paths)
    assert not any("tf_universe_with_family" in path for path in paths)
    assert "generated_at" not in manifest
