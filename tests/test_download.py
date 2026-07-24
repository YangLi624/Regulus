"""Download and extraction contracts for published model bundles."""

from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from regulus.cli import download


def test_download_selects_bundle_manifest_with_nested_graph_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    bundle = source / "fixture-v1"
    graph = bundle / "graph"
    graph.mkdir(parents=True)
    bundle_manifest = {
        "schema_version": "1.0",
        "bundle_id": "fixture-v1",
        "train_config": "config.yaml",
        "perturb_ckpt": "perturb.pt",
        "graph_ckpt": "graph.pt",
        "gene_universe": "genes.csv",
    }
    (bundle / "manifest.json").write_text(
        json.dumps(bundle_manifest), encoding="utf-8"
    )
    (graph / "manifest.json").write_text(
        json.dumps({"asset_id": "graph-v1", "node_counts": {"gene": 2}}),
        encoding="utf-8",
    )
    for filename in ("config.yaml", "perturb.pt", "graph.pt", "genes.csv"):
        (bundle / filename).write_text("fixture", encoding="utf-8")

    archive = tmp_path / "fixture-v1.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in bundle.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(source).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(
        download,
        "_load_catalog",
        lambda: {
            "bundles": {
                "fixture-v1": {
                    "url": "https://example.invalid/fixture-v1.zip",
                    "sha256": digest,
                }
            }
        },
    )

    def fake_urlretrieve(url, destination):
        shutil.copy2(archive, destination)
        return str(destination), None

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_urlretrieve)
    output = tmp_path / "bundles"
    result = download.run_download(
        Namespace(
            bundle_id="fixture-v1",
            output_dir=str(output),
            force=False,
        )
    )
    assert result == 0
    installed = output / "fixture-v1"
    assert json.loads((installed / "manifest.json").read_text())["bundle_id"] == "fixture-v1"
    assert (installed / "graph" / "manifest.json").is_file()
