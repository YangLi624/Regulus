"""Download and verify released Regulus bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from importlib.resources import files
from pathlib import Path


def _load_catalog() -> dict:
    resource = files("regulus").joinpath("assets", "bundles", "catalog.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        root = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Bundle archive contains an unsafe path: {member.filename}")
        handle.extractall(destination)


def _find_bundle_root(extracted: Path) -> Path:
    required_fields = {
        "bundle_id",
        "train_config",
        "perturb_ckpt",
        "graph_ckpt",
        "gene_universe",
    }
    candidates = []
    for manifest_path in extracted.rglob("manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and required_fields.issubset(payload):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise ValueError(
            "Bundle archive must contain exactly one bundle manifest; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def run_download(args) -> int:
    catalog = _load_catalog()
    entries = catalog.get("bundles", {})
    if not args.bundle_id:
        print("Available Regulus bundles:")
        for bundle_id, entry in entries.items():
            print(f"  {bundle_id}: {entry.get('description', '')}")
        return 0

    entry = entries.get(args.bundle_id)
    if entry is None:
        raise KeyError(f"Unknown bundle: {args.bundle_id}")
    url = entry.get("url")
    expected_hash = entry.get("sha256")
    if not url or not expected_hash:
        raise RuntimeError(f"Bundle {args.bundle_id!r} has not been published yet")

    output_root = Path(args.output_dir).expanduser().resolve()
    destination = output_root / args.bundle_id
    if destination.exists() and not args.force:
        raise FileExistsError(f"Bundle already exists: {destination}; pass --force to replace it")
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="regulus-download-") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "bundle.zip"
        urllib.request.urlretrieve(str(url), archive)
        if _sha256(archive) != str(expected_hash):
            raise ValueError(f"SHA-256 mismatch for downloaded bundle {args.bundle_id}")
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        _extract_zip(archive, extracted)
        bundle_root = _find_bundle_root(extracted)
        staged = output_root / f".{args.bundle_id}.staging"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(bundle_root, staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)
    print(f"Installed {args.bundle_id} to {destination}")
    return 0
