"""Trace attribution-supported paths through a frozen Regulus graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from regulus.graph.frozen_hgt import FrozenHGTBundle
from regulus.graph.schema import GENE_CFO, TF_CFO_LLM, TF_GENE

REFERENCE_SUPPORTED = "reference_supported"
LLM_CONTEXT = "llm_context"
EVIDENCE_SOURCES = (REFERENCE_SUPPORTED, LLM_CONTEXT)


@dataclass(frozen=True)
class EvidencePath:
    path_type: str
    candidate: str = ""
    tf: str = ""
    gene: str = ""
    cfo: str = ""
    attribution_score: float = 0.0
    evidence_source: str = REFERENCE_SUPPORTED
    relations: tuple[str, ...] = ()
    graph_edge_weights: tuple[float, ...] = ()


def _as_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _weighted_edges(data, edge_type) -> list[tuple[int, int, float]]:
    if edge_type not in set(data.edge_types):
        return []
    store = data[edge_type]
    edge_index = _as_numpy(store.edge_index)
    weights = _as_numpy(getattr(store, "edge_attr", np.ones(edge_index.shape[1]))).reshape(-1)
    return [
        (int(source), int(target), float(weight))
        for source, target, weight in zip(edge_index[0], edge_index[1], weights)
    ]


class EvidenceGraphBuilder:
    """Expose neutral gene-, CFO-, and candidate-seeded graph traversal."""

    def __init__(
        self,
        hgt_bundle: FrozenHGTBundle,
        gene_symbols: Sequence[str],
        tf_candidate_names: Sequence[str] = (),
        cfo_label_mode: str = "auto",
    ) -> None:
        if not hgt_bundle._loaded:
            hgt_bundle.load()
        if hgt_bundle.data is None:
            raise RuntimeError("Frozen graph data are unavailable")
        self.hgt_bundle = hgt_bundle
        self.data = hgt_bundle.data
        self.gene_symbols = list(map(str, gene_symbols))
        self.gene_symbol_to_idx = {
            symbol: index for index, symbol in enumerate(self.gene_symbols)
        }
        self.tf_candidate_names = list(map(str, tf_candidate_names))
        self.cfo_label_mode = str(cfo_label_mode).lower()
        if self.cfo_label_mode not in {"auto", "id", "name"}:
            raise ValueError("cfo_label_mode must be auto, id, or name")

        self.tf_nodes = hgt_bundle.tf_nodes
        self.cfo_nodes = hgt_bundle.cfo_nodes
        self.tf_symbol_to_idx: dict[str, int] = {}
        self.tf_idx_to_symbol: dict[int, str] = {}
        if self.tf_nodes is not None and "tf_symbol" in self.tf_nodes:
            for row_index, row in self.tf_nodes.iterrows():
                index = int(row.get("tf_index", row_index))
                symbol = str(row["tf_symbol"])
                self.tf_symbol_to_idx[symbol] = index
                self.tf_idx_to_symbol[index] = symbol

        self.cfo_idx_to_id: dict[int, str] = {}
        self.cfo_idx_to_name: dict[int, str] = {}
        if self.cfo_nodes is not None:
            for row_index, row in self.cfo_nodes.iterrows():
                index = int(row.get("cfo_index", row_index))
                if "cfo_id" in row:
                    self.cfo_idx_to_id[index] = str(row["cfo_id"])
                if "cfo_name" in row:
                    self.cfo_idx_to_name[index] = str(row["cfo_name"])

        self.tf_to_gene: dict[int, dict[int, float]] = {}
        self.gene_to_tf: dict[int, dict[int, float]] = {}
        self.gene_to_cfo: dict[int, dict[int, float]] = {}
        self.cfo_to_gene: dict[int, dict[int, float]] = {}
        self.tf_to_cfo_llm: dict[int, dict[int, float]] = {}
        for tf_index, gene_index, weight in _weighted_edges(self.data, TF_GENE):
            self.tf_to_gene.setdefault(tf_index, {})[gene_index] = weight
            self.gene_to_tf.setdefault(gene_index, {})[tf_index] = weight
        for gene_index, cfo_index, weight in _weighted_edges(self.data, GENE_CFO):
            self.gene_to_cfo.setdefault(gene_index, {})[cfo_index] = weight
            self.cfo_to_gene.setdefault(cfo_index, {})[gene_index] = weight
        for tf_index, cfo_index, weight in _weighted_edges(self.data, TF_CFO_LLM):
            self.tf_to_cfo_llm.setdefault(tf_index, {})[cfo_index] = weight

    def _gene_name(self, index: int) -> str:
        return self.gene_symbols[index] if 0 <= index < len(self.gene_symbols) else f"gene_{index}"

    def _cfo_name(self, index: int) -> str:
        cfo_id = self.cfo_idx_to_id.get(index, f"CFO_{index}")
        cfo_name = self.cfo_idx_to_name.get(index)
        if self.cfo_label_mode == "id":
            return cfo_id
        if self.cfo_label_mode == "name" and cfo_name:
            return cfo_name
        return cfo_name or cfo_id

    def _tf_name(self, index: int) -> str:
        return self.tf_idx_to_symbol.get(index, f"TF_{index}")

    def trace_from_genes(
        self,
        gene_indices: Sequence[int],
        attribution_scores: Sequence[float],
        *,
        candidate_names: Optional[Sequence[str]] = None,
        top_n_paths: int = 100,
    ) -> list[EvidencePath]:
        """Trace upstream TF and downstream CFO context for attributed genes."""
        allowed_tf = None
        allowed_genes: set[int] = set()
        if candidate_names is not None:
            allowed_tf = {
                self.tf_symbol_to_idx[name]
                for name in candidate_names
                if name in self.tf_symbol_to_idx
            }
            allowed_genes = {
                self.gene_symbol_to_idx[name]
                for name in candidate_names
                if name in self.gene_symbol_to_idx
            }
        paths: list[EvidencePath] = []
        for gene_index, attribution in zip(gene_indices, attribution_scores):
            gene_index = int(gene_index)
            score = float(attribution)
            tf_edges = self.gene_to_tf.get(gene_index, {})
            cfo_edges = self.gene_to_cfo.get(gene_index, {})
            if candidate_names is None or gene_index in allowed_genes:
                for cfo_index, gene_cfo_weight in cfo_edges.items():
                    paths.append(EvidencePath(
                        path_type="gene_cfo",
                        candidate=(
                            self._gene_name(gene_index)
                            if gene_index in allowed_genes
                            else ""
                        ),
                        gene=self._gene_name(gene_index),
                        cfo=self._cfo_name(cfo_index),
                        attribution_score=score,
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(GENE_CFO[1],),
                        graph_edge_weights=(gene_cfo_weight,),
                    ))
            for tf_index, tf_gene_weight in tf_edges.items():
                if allowed_tf is not None and tf_index not in allowed_tf:
                    continue
                tf_name = self._tf_name(tf_index)
                paths.append(EvidencePath(
                    path_type="tf_gene",
                    candidate=tf_name if allowed_tf is not None else "",
                    tf=tf_name,
                    gene=self._gene_name(gene_index),
                    attribution_score=score,
                    evidence_source=REFERENCE_SUPPORTED,
                    relations=(TF_GENE[1],),
                    graph_edge_weights=(tf_gene_weight,),
                ))
                for cfo_index, gene_cfo_weight in cfo_edges.items():
                    paths.append(EvidencePath(
                        path_type="tf_gene_cfo",
                        candidate=tf_name if allowed_tf is not None else "",
                        tf=tf_name,
                        gene=self._gene_name(gene_index),
                        cfo=self._cfo_name(cfo_index),
                        attribution_score=score,
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(TF_GENE[1], GENE_CFO[1]),
                        graph_edge_weights=(tf_gene_weight, gene_cfo_weight),
                    ))
                    llm_weight = self.tf_to_cfo_llm.get(tf_index, {}).get(cfo_index)
                    if llm_weight is not None:
                        paths.append(EvidencePath(
                            path_type="tf_cfo_context_with_gene_mediator",
                            candidate=tf_name if allowed_tf is not None else "",
                            tf=tf_name,
                            gene=self._gene_name(gene_index),
                            cfo=self._cfo_name(cfo_index),
                            attribution_score=score,
                            evidence_source=LLM_CONTEXT,
                            relations=(TF_GENE[1], GENE_CFO[1], TF_CFO_LLM[1]),
                            graph_edge_weights=(tf_gene_weight, gene_cfo_weight, llm_weight),
                        ))
        paths.sort(key=lambda path: abs(path.attribution_score), reverse=True)
        return paths[:top_n_paths]

    def trace_from_cfos(
        self,
        cfo_indices: Sequence[int],
        attribution_scores: Sequence[float],
        *,
        candidate_names: Optional[Sequence[str]] = None,
        top_n_paths: int = 100,
    ) -> list[EvidencePath]:
        """Trace predicted-driver context for attributed CFO features."""
        candidate_names = list(candidate_names or [])
        allowed_tf = {
            self.tf_symbol_to_idx[name]: name
            for name in candidate_names
            if name in self.tf_symbol_to_idx
        }
        allowed_genes = {
            self.gene_symbol_to_idx[name]: name
            for name in candidate_names
            if name in self.gene_symbol_to_idx
        }
        paths: list[EvidencePath] = []
        for cfo_index, attribution in zip(cfo_indices, attribution_scores):
            cfo_index = int(cfo_index)
            for gene_index, weight in self.cfo_to_gene.get(cfo_index, {}).items():
                if not candidate_names or gene_index in allowed_genes:
                    paths.append(EvidencePath(
                        path_type="gene_cfo",
                        candidate=allowed_genes.get(gene_index, ""),
                        gene=self._gene_name(gene_index),
                        cfo=self._cfo_name(cfo_index),
                        attribution_score=float(attribution),
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(GENE_CFO[1],),
                        graph_edge_weights=(weight,),
                    ))
                for tf_index, tf_gene_weight in self.gene_to_tf.get(gene_index, {}).items():
                    if candidate_names and tf_index not in allowed_tf:
                        continue
                    tf_name = self._tf_name(tf_index)
                    paths.append(EvidencePath(
                        path_type="tf_gene_cfo",
                        candidate=allowed_tf.get(tf_index, ""),
                        tf=tf_name,
                        gene=self._gene_name(gene_index),
                        cfo=self._cfo_name(cfo_index),
                        attribution_score=float(attribution),
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(TF_GENE[1], GENE_CFO[1]),
                        graph_edge_weights=(tf_gene_weight, weight),
                    ))
                    llm_weight = self.tf_to_cfo_llm.get(tf_index, {}).get(cfo_index)
                    if llm_weight is not None:
                        paths.append(EvidencePath(
                            path_type="tf_cfo_context_with_gene_mediator",
                            candidate=allowed_tf.get(tf_index, ""),
                            tf=tf_name,
                            gene=self._gene_name(gene_index),
                            cfo=self._cfo_name(cfo_index),
                            attribution_score=float(attribution),
                            evidence_source=LLM_CONTEXT,
                            relations=(TF_GENE[1], GENE_CFO[1], TF_CFO_LLM[1]),
                            graph_edge_weights=(tf_gene_weight, weight, llm_weight),
                        ))
        paths.sort(key=lambda path: abs(path.attribution_score), reverse=True)
        return paths[:top_n_paths]

    def trace_from_candidates(
        self,
        candidate_names: Sequence[str],
        *,
        top_n_paths: int = 100,
    ) -> list[EvidencePath]:
        """Return candidate graph context without labeling it as attribution evidence."""
        paths: list[EvidencePath] = []
        for name in candidate_names:
            tf_index = self.tf_symbol_to_idx.get(str(name))
            if tf_index is not None:
                for gene_index, weight in self.tf_to_gene.get(tf_index, {}).items():
                    paths.append(EvidencePath(
                        path_type="tf_gene",
                        candidate=str(name),
                        tf=str(name),
                        gene=self._gene_name(gene_index),
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(TF_GENE[1],),
                        graph_edge_weights=(weight,),
                    ))
                continue
            gene_index = self.gene_symbol_to_idx.get(str(name))
            if gene_index is None:
                continue
            for cfo_index, weight in self.gene_to_cfo.get(gene_index, {}).items():
                paths.append(EvidencePath(
                    path_type="driver_gene_cfo",
                    candidate=str(name),
                    gene=str(name),
                    cfo=self._cfo_name(cfo_index),
                    evidence_source=REFERENCE_SUPPORTED,
                    relations=(GENE_CFO[1],),
                    graph_edge_weights=(weight,),
                ))
                for upstream_tf, tf_gene_weight in self.gene_to_tf.get(gene_index, {}).items():
                    paths.append(EvidencePath(
                        path_type="tf_driver_gene_cfo",
                        candidate=str(name),
                        tf=self._tf_name(upstream_tf),
                        gene=str(name),
                        cfo=self._cfo_name(cfo_index),
                        evidence_source=REFERENCE_SUPPORTED,
                        relations=(TF_GENE[1], GENE_CFO[1]),
                        graph_edge_weights=(tf_gene_weight, weight),
                    ))
        return paths[:top_n_paths]

    def build_for_sample(
        self,
        top_gene_indices: Sequence[int],
        top_gene_scores: Sequence[float],
        top_tf_names: Sequence[str],
        top_n_paths: int = 30,
    ) -> list[EvidencePath]:
        """Compatibility wrapper for candidate-filtered gene traversal."""
        return self.trace_from_genes(
            top_gene_indices,
            top_gene_scores,
            candidate_names=top_tf_names,
            top_n_paths=top_n_paths,
        )

    @staticmethod
    def to_jsonable(paths: Iterable[EvidencePath]) -> list[dict[str, object]]:
        rows = []
        for path in paths:
            row = asdict(path)
            row["relations"] = list(path.relations)
            row["graph_edge_weights"] = list(path.graph_edge_weights)
            rows.append(row)
        return rows
