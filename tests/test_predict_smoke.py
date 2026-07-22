"""Bundle schema and optional end-to-end prediction smoke tests."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BUNDLE = (
    Path(os.environ["REGULUS_SMOKE_BUNDLE"])
    if "REGULUS_SMOKE_BUNDLE" in os.environ
    else None
)
H5AD = (
    Path(os.environ["REGULUS_SMOKE_H5AD"])
    if "REGULUS_SMOKE_H5AD" in os.environ
    else None
)


def _assets_ready() -> bool:
    if BUNDLE is None or H5AD is None:
        return False
    manifest_file = BUNDLE / "manifest.json"
    if not manifest_file.exists() or not H5AD.exists():
        return False
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return all(
        (BUNDLE / manifest[field]).exists()
        for field in ("perturb_ckpt", "graph_ckpt", "train_config", "gene_universe")
    )


def test_bundle_catalog_lists_only_release_models():
    catalog_path = ROOT / "regulus/assets/bundles/catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert set(catalog["bundles"]) == {
        "norman-post-state-v1",
        "schmidt-post-state-v1",
        "tian-crispra-post-state-v1",
        "joung-tfatlas-cfo-post-state-v1",
    }
    for entry in catalog["bundles"].values():
        assert entry["representation"] == "post_state"
        assert entry["channels"] in {"gene_cfo", "cfo"}


@pytest.mark.skipif(not _assets_ready(), reason="checkpoint or H5AD not present")
def test_regulus_predict_smoke(tmp_path):
    from regulus import RegulusModel

    assert BUNDLE is not None and H5AD is not None
    model = RegulusModel.from_bundle(BUNDLE)
    frame = model.predict(H5AD, mode="joint", top_k=3)
    assert {"cell_id", "perturbation", "score"}.issubset(frame.columns)
    output = tmp_path / "predictions.csv"
    frame.to_csv(output, index=False)
    assert output.stat().st_size > 0
