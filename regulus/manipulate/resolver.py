"""Resolve CFO identifiers and names to frozen graph indices."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from regulus.graph.frozen_hgt import FrozenHGTBundle


def _detect_col(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((column for column in candidates if column in frame.columns), None)


def build_cfo_lookup(hgt_bundle: FrozenHGTBundle) -> dict[str, tuple[int, ...]]:
    """Map CFO IDs and names to one or more graph indices."""
    if not hgt_bundle._loaded:
        hgt_bundle.load()
    nodes = hgt_bundle.cfo_nodes
    if nodes is None:
        return {}
    index_column = _detect_col(nodes, ("cfo_index", "index", "id"))
    id_column = _detect_col(nodes, ("cfo_id",))
    name_column = _detect_col(nodes, ("cfo_name", "name"))
    values: dict[str, set[int]] = {}
    for row_index, row in nodes.iterrows():
        cfo_index = int(row[index_column]) if index_column else int(row_index)
        keys = [f"cfo_{cfo_index}"]
        if id_column:
            keys.append(str(row[id_column]).strip())
        if name_column:
            keys.append(str(row[name_column]).strip())
        for key in keys:
            values.setdefault(key, set()).add(cfo_index)
    return {key: tuple(sorted(indices)) for key, indices in values.items()}


def resolve_cfo_targets(
    targets: Sequence[str],
    deltas: Sequence[float],
    hgt_bundle: FrozenHGTBundle,
) -> tuple[list[int], list[float]]:
    lookup = build_cfo_lookup(hgt_bundle)
    if not lookup:
        raise RuntimeError("CFO node metadata are unavailable in the graph asset")
    if len(deltas) == 1 and len(targets) > 1:
        deltas = [float(deltas[0])] * len(targets)
    if len(targets) != len(deltas):
        raise ValueError("cfo_targets and cfo_delta must have equal lengths")

    indices: list[int] = []
    applied: list[float] = []
    for target, delta in zip(targets, deltas):
        key = str(target).strip()
        matches = lookup.get(key)
        if matches is None:
            raise KeyError(f"CFO {key!r} was not found in the graph asset")
        if len(matches) != 1:
            raise ValueError(f"CFO label {key!r} is ambiguous; use its CFO identifier")
        indices.append(matches[0])
        applied.append(float(delta))
    return indices, applied
