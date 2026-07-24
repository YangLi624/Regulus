"""Opt-in golden regression tests for the four published model bundles."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
import torch

from regulus import RegulusModel
from regulus.io.bundle import sha256_file


BUNDLE_ROOT_VALUE = os.environ.get("REGULUS_BUNDLE_ROOT")
EXAMPLE_H5AD_VALUE = os.environ.get(
    "REGULUS_GOLDEN_INPUT", "examples/data/example_cells.regulus.h5ad"
)
GOLDEN_PATH = Path(__file__).parent / "golden" / "release_bundles_v1.json"

pytestmark = pytest.mark.skipif(
    not BUNDLE_ROOT_VALUE,
    reason="set REGULUS_BUNDLE_ROOT to run published-bundle regression tests",
)


def _update_matrix_digest(digest, matrix) -> None:
    if sparse.issparse(matrix):
        value = matrix.tocsr(copy=True)
        value.sort_indices()
        digest.update(b"csr")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.indptr.tobytes())
        digest.update(value.indices.tobytes())
        digest.update(value.data.tobytes())
    else:
        value = np.ascontiguousarray(matrix)
        digest.update(b"dense")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())


def _semantic_h5ad_sha256(path: Path) -> str:
    adata = ad.read_h5ad(path)
    digest = hashlib.sha256()
    _update_matrix_digest(digest, adata.X)
    _update_matrix_digest(digest, adata.obsm["X_cfo_activity"])
    for values in (
        adata.obs_names.astype(str),
        adata.var_names.astype(str),
        np.asarray(adata.uns["regulus_cfo_ids"]).astype(str),
    ):
        digest.update("\n".join(values).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _prediction_arrays(frame):
    identity = [
        (str(row.cell_id), int(row.rank), str(row.perturbation), str(row.mode))
        for row in frame.itertuples(index=False)
    ]
    scores = np.asarray([float(row.score) for row in frame.itertuples(index=False)])
    return identity, scores


def _golden_prediction_arrays(records):
    identity = [
        (
            str(row["cell_id"]),
            int(row["rank"]),
            str(row["candidate"]),
            str(row["mode"]),
        )
        for row in records
    ]
    scores = np.asarray([float(row["score"]) for row in records])
    return identity, scores


def test_published_bundle_golden_outputs():
    bundle_root = Path(BUNDLE_ROOT_VALUE).resolve()
    input_h5ad = Path(EXAMPLE_H5AD_VALUE).resolve()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert _semantic_h5ad_sha256(input_h5ad) == golden["input_semantic_sha256"]

    device = os.environ.get(
        "REGULUS_GOLDEN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    atol = float(golden["score_atol"])
    for expected in golden["bundles"]:
        bundle_path = bundle_root / expected["bundle_id"]
        assert sha256_file(bundle_path / "manifest.json") == expected["manifest_sha256"]
        model = RegulusModel.from_bundle(bundle_path, device=device)
        assert sha256_file(model.manifest.perturb_ckpt) == expected[
            "perturb_checkpoint_sha256"
        ]
        assert len(model.candidate_perturbations) == expected["candidate_count"]
        assert list(model.manifest.supported_modes) == expected["supported_modes"]
        assert model.manifest.default_mode == expected["default_mode"]

        for mode, expected_rows in expected["predictions"].items():
            observed = model.predict(
                input_h5ad, mode=mode, top_k=int(golden["top_k"]), batch_size=8
            )
            observed_id, observed_scores = _prediction_arrays(observed)
            expected_id, expected_scores = _golden_prediction_arrays(expected_rows)
            assert observed_id == expected_id
            np.testing.assert_allclose(observed_scores, expected_scores, rtol=1e-5, atol=atol)

        if "manipulation" in expected:
            observed = model.manipulate(
                cfo_targets=["cfo_0"],
                cfo_delta=[0.1],
                anchor_h5ad=input_h5ad,
                mode="cfo_only",
                top_k=int(golden["top_k"]),
                sort_by="rank_gain",
            )
            expected_rows = expected["manipulation"]
            assert [
                (
                    int(row.rank),
                    str(row.perturbation),
                    int(row.rank_before),
                    int(row.rank_after),
                    int(row.rank_gain),
                )
                for row in observed.itertuples(index=False)
            ] == [
                (
                    int(row["rank"]),
                    str(row["candidate"]),
                    int(row["rank_before"]),
                    int(row["rank_after"]),
                    int(row["rank_gain"]),
                )
                for row in expected_rows
            ]
            for column in ("score_before", "score_after", "score_delta"):
                np.testing.assert_allclose(
                    observed[column].to_numpy(),
                    np.asarray([float(row[column]) for row in expected_rows]),
                    rtol=1e-5,
                    atol=atol,
                )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
