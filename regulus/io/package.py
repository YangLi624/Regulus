"""Assemble deterministic, self-contained Regulus release bundles."""

from __future__ import annotations

from copy import deepcopy
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from regulus.graph.assets import validate_graph_asset
from regulus.io.bundle import BUNDLE_SCHEMA_VERSION, sha256_file
from regulus.perturb.spec import PerturbModelSpec, normalize_mode
from regulus.utils.config import load_config


def _copy_file(source: str | Path, destination: Path) -> None:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _portable_config(config: dict) -> dict:
    portable = deepcopy(config)
    data = portable.setdefault("data", {})
    data["dataset_dir"] = "data"
    data["graph_asset_dir"] = "graph"
    data["gene_universe"] = "graph/universes/gene_universe.csv"
    model = portable.setdefault("model", {})
    model["graph_checkpoint"] = "checkpoints/graph_model.pt"
    portable["output_dir"] = "outputs"
    portable["checkpoint_dir"] = "outputs/checkpoints"
    portable["resume"] = False
    return portable


def _file_index(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" and path.parent == root:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def _write_deterministic_zip(source_root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
            relative = path.relative_to(source_root.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes(), compresslevel=9)


def build_release_bundle(
    *,
    bundle_id: str,
    config_path: str | Path,
    perturb_checkpoint: str | Path,
    graph_checkpoint: str | Path,
    graph_asset_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build one portable bundle archive and return its path."""
    config_path = Path(config_path).expanduser().resolve()
    graph_asset_dir = Path(graph_asset_dir).expanduser().resolve()
    validate_graph_asset(graph_asset_dir, validate_hashes=True)
    config = load_config(config_path)
    spec = PerturbModelSpec.from_config(config)
    supported_modes = {
        "gene": ["gene_only"],
        "cfo": ["cfo_only"],
        "gene_cfo": ["joint", "gene_only", "cfo_only"],
    }[spec.channels]
    default_mode = normalize_mode(None, spec.channels)

    with tempfile.TemporaryDirectory(prefix="regulus-bundle-") as temporary:
        root = Path(temporary) / bundle_id
        root.mkdir()
        bundle_config = root / "config" / "perturb_config.yaml"
        bundle_config.parent.mkdir(parents=True, exist_ok=True)
        bundle_config.write_text(
            yaml.safe_dump(_portable_config(config), sort_keys=False), encoding="utf-8"
        )
        _copy_file(perturb_checkpoint, root / "checkpoints" / "perturb_model.pt")
        _copy_file(graph_checkpoint, root / "checkpoints" / "graph_model.pt")
        shutil.copytree(
            graph_asset_dir,
            root / "graph",
            ignore=shutil.ignore_patterns("cache", "__pycache__", "*.pyc"),
        )
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            **spec.as_dict(),
            "supported_modes": supported_modes,
            "default_mode": default_mode,
            "train_config": "config/perturb_config.yaml",
            "perturb_ckpt": "checkpoints/perturb_model.pt",
            "graph_ckpt": "checkpoints/graph_model.pt",
            "graph_asset": "graph/manifest.json",
            "gene_universe": "graph/universes/gene_universe.csv",
            "files": _file_index(root),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        archive = Path(output_dir).expanduser().resolve() / f"{bundle_id}.zip"
        _write_deterministic_zip(root, archive)
    return archive
