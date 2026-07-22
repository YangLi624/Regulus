"""Tests for pySCENIC expression export (HGNC symbol rows)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

_BASELINE = Path(__file__).resolve().parents[1] / "tasks" / "03_inverse_baselines"
_setup = _BASELINE / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("_ib_path_setup", _setup)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.install()

from common.scenic_export import (  # noqa: E402
    adata_to_pyscenic_csv,
    collapse_expression_by_symbol,
    count_tf_overlap,
)


def test_collapse_ensembl_index_to_gene_symbols(tmp_path: Path) -> None:
    """Replogle/Tian-style h5ad: Ensembl var index + gene_name symbols."""
    var = pd.DataFrame(
        {"gene_name": ["MYC", "TP53", "MYC"]},
        index=pd.Index(
            ["ENSG00000136997", "ENSG00000141510", "ENSG00000136998"],
            name="gene_name",
        ),
    )
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=["cell0"]),
        var=var,
    )
    matrix, genes = collapse_expression_by_symbol(adata)
    assert genes == ["MYC", "TP53"]
    assert matrix.shape == (1, 2)
    assert abs(matrix[0, 0] - 2.0) < 1e-6  # mean of 1 and 3

    out = tmp_path / "expr.csv"
    adata_to_pyscenic_csv(adata, out)
    df = pd.read_csv(out, index_col=0)
    assert "MYC" in df.index
    assert "ENSG00000136997" not in df.index


def test_tf_overlap_counts_symbols(tmp_path: Path) -> None:
    tfs = tmp_path / "tfs.txt"
    tfs.write_text("MYC\nTP53\n", encoding="utf-8")
    n, overlap = count_tf_overlap(["MYC", "BRCA1"], tfs)
    assert n == 1
    assert overlap == {"MYC"}
