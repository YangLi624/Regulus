"""Build model-ready H5AD datasets from prepared matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from regulus.perturb.data import PerturbationDataset


def build_predict_dataset(
    h5ad_path: str | Path,
    gene_universe_path: str | Path,
    *,
    representation: str = "post_state",
    channels: str = "gene_cfo",
    expected_cfo_ids: Optional[Sequence[str]] = None,
    include_controls: bool = True,
) -> PerturbationDataset:
    """Read prepared matrices without deriving baselines or dropping rows."""
    return PerturbationDataset(
        h5ad_path,
        gene_universe_path,
        representation=representation,
        channels=channels,
        expected_cfo_ids=expected_cfo_ids,
        require_labels=False,
        include_controls=include_controls,
    )
