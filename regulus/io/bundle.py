"""Portable Regulus model-bundle manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from regulus.perturb.spec import normalize_channels, normalize_head, normalize_representation

BUNDLE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BundleManifest:
    bundle_id: str
    train_config: Path
    perturb_ckpt: Path
    graph_ckpt: Path
    gene_universe: Path
    representation: str
    channels: str
    head: str
    supported_modes: Sequence[str] = field(default_factory=tuple)
    default_mode: str = "joint"
    graph_asset_dir: Optional[Path] = None
    bundle_dir: Path = field(default_factory=Path)
    schema_version: str = BUNDLE_SCHEMA_VERSION
    graph_asset_id: Optional[str] = None

    def validate_input_representation(self, requested: Optional[str]) -> str:
        if requested is not None and normalize_representation(requested) != self.representation:
            raise ValueError(
                f"requested representation {requested!r} does not match bundle "
                f"representation {self.representation!r}"
            )
        return self.representation


def resolve_bundle_path(path: str | Path) -> Path:
    """Resolve a user-provided bundle path without repository-relative fallback."""
    value = Path(path).expanduser().resolve()
    if not value.exists():
        raise FileNotFoundError(f"Bundle path not found: {value}")
    return value


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bundle_file(bundle_dir: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bundle field {field_name!r} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"Bundle field {field_name!r} must be relative to the bundle root: {value!r}"
        )
    resolved = (bundle_dir / relative).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Bundle field {field_name!r} not found: {resolved}")
    return resolved


def _verify_files(raw: dict[str, Any], bundle_dir: Path) -> None:
    for entry in raw.get("files", []):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Each bundle files entry must contain a path")
        path = _resolve_bundle_file(bundle_dir, entry["path"], "files.path")
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(f"Bundle file size mismatch: {entry['path']}")
        expected_hash = entry.get("sha256")
        if expected_hash is not None and sha256_file(path) != str(expected_hash):
            raise ValueError(f"Bundle file SHA-256 mismatch: {entry['path']}")


def load_bundle_manifest(bundle_path: str | Path) -> BundleManifest:
    path = resolve_bundle_path(bundle_path)
    manifest_file = path if path.name == "manifest.json" else path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_file}")
    bundle_dir = manifest_file.parent.resolve()
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Bundle manifest root must be a JSON object")

    schema_version = str(raw.get("schema_version", BUNDLE_SCHEMA_VERSION))
    if schema_version.split(".", 1)[0] != BUNDLE_SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(f"Unsupported bundle schema version: {schema_version}")

    required = {
        name: _resolve_bundle_file(bundle_dir, raw.get(name), name)
        for name in ("train_config", "perturb_ckpt", "graph_ckpt", "gene_universe")
    }
    graph_asset = None
    if raw.get("graph_asset") is not None:
        graph_asset_path = _resolve_bundle_file(bundle_dir, raw["graph_asset"], "graph_asset")
        graph_asset = (
            graph_asset_path.parent
            if graph_asset_path.name == "manifest.json"
            else graph_asset_path
        )

    config_raw = yaml.safe_load(required["train_config"].read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        raise ValueError("Bundle train_config root must be a YAML mapping")
    model = config_raw.get("model", {})
    representation = normalize_representation(
        str(raw.get("representation", model.get("representation", "post_state")))
    )
    channels = normalize_channels(str(raw.get("channels", model.get("channels", "gene_cfo"))))
    head = normalize_head(str(raw.get("head", model.get("head", "prototype_matching"))))

    modes = raw.get("supported_modes") or {
        "gene": ["gene_only"],
        "cfo": ["cfo_only"],
        "gene_cfo": ["joint", "gene_only", "cfo_only"],
    }[channels]
    normalized_modes = tuple(str(mode) for mode in modes)
    raw_default = raw.get("default_mode", normalized_modes[0])
    default_mode = str(raw_default)
    if default_mode not in normalized_modes:
        raise ValueError("Bundle default_mode must be listed in supported_modes")

    _verify_files(raw, bundle_dir)
    graph_asset_id = raw.get("graph_asset_id")
    if graph_asset is not None and (graph_asset / "manifest.json").is_file():
        graph_manifest = json.loads((graph_asset / "manifest.json").read_text(encoding="utf-8"))
        graph_asset_id = graph_asset_id or graph_manifest.get("asset_id")

    return BundleManifest(
        bundle_id=str(raw.get("bundle_id", bundle_dir.name)),
        train_config=required["train_config"],
        perturb_ckpt=required["perturb_ckpt"],
        graph_ckpt=required["graph_ckpt"],
        gene_universe=required["gene_universe"],
        representation=representation,
        channels=channels,
        head=head,
        supported_modes=normalized_modes,
        default_mode=default_mode,
        graph_asset_dir=graph_asset,
        bundle_dir=bundle_dir,
        schema_version=schema_version,
        graph_asset_id=None if graph_asset_id is None else str(graph_asset_id),
    )
