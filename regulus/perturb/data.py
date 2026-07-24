"""H5AD input dataset for perturbation training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset

from regulus.perturb.representation import subtract_control_mean, validate_input_matrix
from regulus.perturb.spec import normalize_channels, normalize_representation
from regulus.preprocess.ucell import require_cfo_activity_matrix, validate_cfo_activity


class PerturbationDataset(Dataset):
    """Aligned gene/CFO inputs for post-state or dataset-derived delta models."""

    def __init__(
        self,
        h5ad_path: str | Path,
        gene_universe_path: str | Path,
        *,
        representation: str,
        channels: str,
        candidate_to_index: Optional[Dict[str, int]] = None,
        require_labels: bool = False,
        include_controls: bool = False,
        expected_n_cfo: Optional[int] = None,
        expected_cfo_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.h5ad_path = Path(h5ad_path)
        self.representation = normalize_representation(representation)
        self.channels = normalize_channels(channels)
        self.candidate_to_index = candidate_to_index
        self.require_labels = bool(require_labels)

        if not self.h5ad_path.exists():
            raise FileNotFoundError(f"h5ad not found: {self.h5ad_path}")
        universe = pd.read_csv(gene_universe_path)
        if "gene_symbol" not in universe:
            raise ValueError("gene_universe.csv requires a gene_symbol column")
        self.gene_symbols = universe["gene_symbol"].astype(str).tolist()
        self.gene_to_index = {gene: i for i, gene in enumerate(self.gene_symbols)}

        adata = sc.read_h5ad(self.h5ad_path)
        first_position: Dict[str, int] = {}
        for i, gene in enumerate(adata.var_names.astype(str)):
            first_position.setdefault(gene, i)
        present = [gene for gene in self.gene_symbols if gene in first_position]
        positions = [first_position[gene] for gene in present]
        self.adata = adata[:, positions].copy()
        self._aligned_positions = np.asarray(
            [self.gene_to_index[gene] for gene in present], dtype=np.int64
        )

        has_labels = "perturbation" in self.adata.obs
        if require_labels and not has_labels:
            raise ValueError("training h5ad requires obs['perturbation']")
        if self.representation == "delta" and not has_labels:
            raise ValueError(
                "representation='delta' requires obs['perturbation'] with rows labelled "
                "'control' so Regulus can calculate the dataset control mean"
            )

        labels = self.adata.obs["perturbation"].astype(str) if has_labels else None
        self.control_rows = (
            np.flatnonzero(labels.to_numpy() == "control").tolist()
            if labels is not None
            else []
        )
        if self.representation == "delta" and not self.control_rows:
            raise ValueError(
                "representation='delta' requires at least one row with "
                "obs['perturbation'] == 'control'"
            )

        if labels is not None and not include_controls:
            self.row_indices = np.flatnonzero(labels.to_numpy() != "control").tolist()
        else:
            self.row_indices = list(range(self.adata.n_obs))

        self.gene_control_mean: Optional[np.ndarray] = None
        if self.representation == "delta" and self.channels in ("gene", "gene_cfo"):
            values = self.adata.X[np.asarray(self.control_rows, dtype=np.int64)].mean(axis=0)
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            aligned = np.zeros(len(self.gene_symbols), dtype=np.float32)
            aligned[self._aligned_positions] = values
            self.gene_control_mean = validate_input_matrix(
                aligned, "delta", name="gene control mean"
            )

        self.cfo_matrix: Optional[np.ndarray] = None
        self.cfo_control_mean: Optional[np.ndarray] = None
        if self.channels in ("cfo", "gene_cfo"):
            self.cfo_matrix = validate_input_matrix(
                require_cfo_activity_matrix(self.adata, context=str(self.h5ad_path)),
                self.representation,
                name="CFO matrix",
            )
            expected = expected_cfo_ids if expected_cfo_ids is not None else expected_n_cfo
            if expected is not None:
                validate_cfo_activity(self.adata, expected, context=str(self.h5ad_path))
            if self.representation == "delta":
                self.cfo_control_mean = validate_input_matrix(
                    self.cfo_matrix[np.asarray(self.control_rows, dtype=np.int64)].mean(axis=0),
                    "delta",
                    name="CFO control mean",
                )

        # These views let filtering operate without copying the full AnnData object.
        key = str(self.h5ad_path)
        self.hca = [key]
        self.perturb_rows = {key: self.row_indices}
        self.cell_names = {
            key: [str(self.adata.obs_names[i]) for i in self.row_indices]
        }
        self.reader = type("ReaderView", (), {})()
        self.reader.data = {key: {"hca": self.adata}}

    def __len__(self) -> int:
        return len(self.row_indices)

    def _gene_row(self, row: int) -> np.ndarray:
        values = self.adata.X[row]
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        aligned = np.zeros(len(self.gene_symbols), dtype=np.float32)
        aligned[self._aligned_positions] = values
        aligned = validate_input_matrix(aligned, self.representation, name="gene matrix")
        if self.representation == "delta":
            assert self.gene_control_mean is not None
            return subtract_control_mean(
                aligned, self.gene_control_mean, name="gene matrix"
            )
        return aligned

    def __getitem__(self, index: int) -> dict:
        row = int(self.row_indices[index])
        sample: dict = {
            "cell_id": str(self.adata.obs_names[row]),
            "representation": self.representation,
        }
        if self.channels in ("gene", "gene_cfo"):
            gene = torch.from_numpy(self._gene_row(row))
            sample["gene_input"] = gene
        if self.cfo_matrix is not None:
            cfo_values = np.asarray(self.cfo_matrix[row], dtype=np.float32)
            if self.representation == "delta":
                assert self.cfo_control_mean is not None
                cfo_values = subtract_control_mean(
                    cfo_values, self.cfo_control_mean, name="CFO matrix"
                )
            cfo = torch.from_numpy(cfo_values)
            sample["cfo_input"] = cfo

        if "perturbation" in self.adata.obs:
            label = str(self.adata.obs["perturbation"].iloc[row])
            sample["perturb_label"] = label
            if self.candidate_to_index is not None:
                sample["label"] = int(self.candidate_to_index.get(label, -1))
        return sample

    def get_all_perturbations(self) -> list[str]:
        if "perturbation" not in self.adata.obs:
            return []
        values = self.adata.obs["perturbation"].astype(str).iloc[self.row_indices]
        return sorted(set(values))
