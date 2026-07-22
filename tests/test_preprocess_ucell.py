"""CFO activity preprocessing contracts."""

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from regulus.preprocess.ucell import (
    CFO_ACTIVITY_OBSM_KEY,
    REGULUS_CFO_IDS_UNS_KEY,
    compute_ucell_cfo_activity,
    load_cfo_gene_sets,
    missing_cfo_activity_message,
    require_cfo_activity_matrix,
    validate_cfo_activity,
)


def test_load_cfo_gene_sets_preserves_order(tmp_path: Path):
    path = tmp_path / "cfo_sets.csv"
    pd.DataFrame({
        "cfo_id": ["GO:0000002", "GO:0000001"],
        "genes": ["C", "A;B"],
    }).to_csv(path, index=False)
    sets = load_cfo_gene_sets(path)
    assert list(sets) == ["GO:0000002", "GO:0000001"]
    assert sets["GO:0000001"] == ["A", "B"]


def test_missing_activity_message_mentions_preprocess():
    message = missing_cfo_activity_message()
    assert "regulus preprocess" in message
    assert CFO_ACTIVITY_OBSM_KEY in message


def test_missing_activity_key_is_rejected():
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(X=np.zeros((2, 3)))
    with pytest.raises(KeyError, match=CFO_ACTIVITY_OBSM_KEY):
        require_cfo_activity_matrix(adata)


def test_validate_cfo_activity_checks_identifier_order():
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.obsm[CFO_ACTIVITY_OBSM_KEY] = np.zeros((2, 2), dtype=np.float32)
    adata.uns[REGULUS_CFO_IDS_UNS_KEY] = ["GO:1", "GO:2"]
    validate_cfo_activity(adata, ["GO:1", "GO:2"])
    with pytest.raises(ValueError, match="order"):
        validate_cfo_activity(adata, ["GO:2", "GO:1"])


def test_compute_ucell_writes_compact_matrix_and_removes_columns(monkeypatch):
    ad = pytest.importorskip("anndata")
    adata = ad.AnnData(X=np.zeros((2, 3)))
    adata.var_names = ["A", "B", "C"]

    def fake_compute(adata_arg, signatures, suffix="_UCell", n_jobs=1):
        adata_arg.obs["GO_0000001_UCell"] = [0.3, 0.4]
        adata_arg.obs["GO_0000002_UCell"] = [0.5, 0.6]

    monkeypatch.setitem(
        sys.modules,
        "pyucell",
        types.SimpleNamespace(compute_ucell_scores=fake_compute),
    )
    cfo_ids = compute_ucell_cfo_activity(
        adata,
        {"GO:0000001": ["A", "B"], "GO:0000002": ["C"]},
    )
    assert cfo_ids == ["GO:0000001", "GO:0000002"]
    assert not any("UCell" in column for column in adata.obs.columns)
    np.testing.assert_allclose(
        adata.obsm[CFO_ACTIVITY_OBSM_KEY],
        np.array([[0.3, 0.5], [0.4, 0.6]], dtype=np.float32),
    )


def test_layer_preprocessing_does_not_replace_saved_x(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    from regulus.preprocess import ucell

    original = np.array([[1.0, 2.0]], dtype=np.float32)
    adata = ad.AnnData(X=original.copy())
    adata.var_names = ["A", "B"]
    adata.layers["counts"] = np.array([[10.0, 20.0]], dtype=np.float32)
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    sets_path = tmp_path / "sets.csv"
    adata.write_h5ad(input_path)
    pd.DataFrame({"cfo_id": ["GO:1"], "genes": ["A;B"]}).to_csv(
        sets_path, index=False
    )

    def fake_compute(adata_arg, gene_sets):
        np.testing.assert_array_equal(adata_arg.X, adata_arg.layers["counts"])
        adata_arg.obsm[CFO_ACTIVITY_OBSM_KEY] = np.ones((1, 1), dtype=np.float32)
        adata_arg.uns[REGULUS_CFO_IDS_UNS_KEY] = list(gene_sets)
        return list(gene_sets)

    monkeypatch.setattr(ucell, "compute_ucell_cfo_activity", fake_compute)
    ucell.preprocess_h5ad_ucell(
        input_path,
        output_path,
        cfo_gene_sets=sets_path,
        layer="counts",
    )
    saved = ad.read_h5ad(output_path)
    np.testing.assert_array_equal(saved.X, original)
