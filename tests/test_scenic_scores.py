"""Unit tests for SCENIC inverse scoring helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_BASELINE = Path(__file__).resolve().parents[1] / "tasks" / "03_inverse_baselines"
_setup = _BASELINE / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("_ib_path_setup", _setup)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
_mod.install()

from common.scenic_scores import (  # noqa: E402
    build_tf_to_regulon_columns,
    rank_candidates_for_cell,
    regulon_tf_name,
    score_candidate_from_aucell_row,
)


def test_regulon_tf_name_parsing() -> None:
    assert regulon_tf_name("MYC(+)") == "MYC"
    assert regulon_tf_name("CEBPB(-)") == "CEBPB"


def test_rank_single_and_multi_gene_candidates() -> None:
    row = pd.Series({"MYC(+)": 0.9, "MYC(-)": 0.1, "TP53(+)": 0.2})
    tf_to_cols = build_tf_to_regulon_columns(row.index)

    assert score_candidate_from_aucell_row(row, "MYC", tf_to_cols) == 0.9
    assert score_candidate_from_aucell_row(row, "TP53", tf_to_cols) == 0.2

    ranked = rank_candidates_for_cell(row, ["TP53", "MYC"], tf_to_cols)
    assert ranked[0][0] == "MYC"
    assert ranked[1][0] == "TP53"

    combo = score_candidate_from_aucell_row(
        row, "MYC+TP53", tf_to_cols, multi_gene_agg="mean"
    )
    assert abs(combo - (0.9 + 0.2) / 2) < 1e-6
