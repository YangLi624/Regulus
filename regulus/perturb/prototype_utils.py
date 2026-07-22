"""Build candidate perturbation prototypes from frozen graph embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from regulus.graph.frozen_hgt import FrozenHGTBundle


def load_gene_symbol_to_idx(
    gene_universe_path: str | Path,
    graph_asset_dir: str | Path | None = None,
) -> Dict[str, int]:
    """Map gene symbols to rows in the graph gene embedding table."""
    path = Path(gene_universe_path)
    if path.exists():
        frame = pd.read_csv(path)
        if "gene_symbol" not in frame.columns:
            raise ValueError(f"{path} does not contain a gene_symbol column")
        if "gene_index" in frame.columns:
            return {
                str(symbol): int(index)
                for symbol, index in zip(frame["gene_symbol"], frame["gene_index"])
            }
        return {
            str(symbol): index
            for index, symbol in enumerate(frame["gene_symbol"].astype(str))
        }

    if graph_asset_dir is not None:
        order_path = Path(graph_asset_dir) / "embeddings" / "gene_order.txt"
        if order_path.exists():
            symbols = [
                line.strip()
                for line in order_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return {symbol: index for index, symbol in enumerate(symbols)}

    raise FileNotFoundError(
        f"Cannot resolve gene indices from {path} or graph embeddings/gene_order.txt"
    )


def load_tf_symbol_to_idx(
    hgt: FrozenHGTBundle,
    gene_universe_path: str | Path,
    graph_asset_dir: str | Path | None = None,
) -> Dict[str, int]:
    """Map TF symbols to rows in the graph TF embedding table."""
    if hgt.tf_nodes is not None and "tf_symbol" in hgt.tf_nodes.columns:
        index_column = "tf_index" if "tf_index" in hgt.tf_nodes.columns else None
        if index_column:
            return {
                str(symbol): int(index)
                for symbol, index in zip(
                    hgt.tf_nodes["tf_symbol"], hgt.tf_nodes[index_column]
                )
            }
        return {
            str(row["tf_symbol"]): index
            for index, (_, row) in enumerate(hgt.tf_nodes.iterrows())
        }

    tf_path = Path(gene_universe_path).parent / "tf_universe.csv"
    if tf_path.exists():
        frame = pd.read_csv(tf_path)
        if "tf_symbol" not in frame.columns:
            raise ValueError(f"{tf_path} does not contain a tf_symbol column")
        if "tf_index" in frame.columns:
            return {
                str(symbol): int(index)
                for symbol, index in zip(frame["tf_symbol"], frame["tf_index"])
            }
        return {
            str(symbol): index
            for index, symbol in enumerate(frame["tf_symbol"].astype(str))
        }

    if graph_asset_dir is not None:
        node_path = Path(graph_asset_dir) / "nodes" / "nodes_tf.csv"
        if node_path.exists():
            frame = pd.read_csv(node_path)
            if "tf_symbol" in frame.columns:
                if "tf_index" in frame.columns:
                    return {
                        str(symbol): int(index)
                        for symbol, index in zip(frame["tf_symbol"], frame["tf_index"])
                    }
                return {
                    str(symbol): index
                    for index, symbol in enumerate(frame["tf_symbol"].astype(str))
                }

    raise FileNotFoundError(
        "Cannot resolve TF indices from the graph node table or tf_universe.csv"
    )


def build_candidate_prototypes(
    candidate_perturbations: Sequence[str],
    candidate_is_tf: Dict[str, bool],
    hgt: FrozenHGTBundle,
    gene_universe_path: str | Path,
    *,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Return frozen TF or gene prototypes in candidate order."""
    if hgt.node_embeddings is None:
        raise RuntimeError("FrozenHGTBundle has not loaded node embeddings")

    h_tf = hgt.node_embeddings.get("tf")
    h_gene = hgt.node_embeddings.get("gene")
    if h_tf is None or h_gene is None:
        raise RuntimeError("Prototype matching requires TF and gene graph embeddings")

    target_device = device or h_tf.device
    h_tf = h_tf.to(target_device)
    h_gene = h_gene.to(target_device)

    graph_asset_dir = Path(gene_universe_path).parent.parent
    gene_indices = load_gene_symbol_to_idx(gene_universe_path, graph_asset_dir)
    tf_indices = load_tf_symbol_to_idx(hgt, gene_universe_path, graph_asset_dir)

    rows: List[torch.Tensor] = []
    kinds: List[str] = []
    missing_tf: List[str] = []
    missing_gene: List[str] = []

    for perturbation in candidate_perturbations:
        name = str(perturbation)
        if candidate_is_tf.get(name, False):
            index = tf_indices.get(name)
            if index is None:
                missing_tf.append(name)
                continue
            if not 0 <= index < h_tf.shape[0]:
                raise IndexError(
                    f"TF {name!r} has index {index}, outside {h_tf.shape[0]} TF rows"
                )
            rows.append(h_tf[index])
            kinds.append("tf")
        else:
            index = gene_indices.get(name)
            if index is None:
                missing_gene.append(name)
                continue
            if not 0 <= index < h_gene.shape[0]:
                raise IndexError(
                    f"Gene {name!r} has index {index}, outside {h_gene.shape[0]} gene rows"
                )
            rows.append(h_gene[index])
            kinds.append("gene")

    if missing_tf:
        raise KeyError(
            f"{len(missing_tf)} TF candidates are absent from the TF universe; "
            f"examples: {missing_tf[:5]}"
        )
    if missing_gene:
        raise KeyError(
            f"{len(missing_gene)} gene candidates are absent from the gene universe; "
            f"examples: {missing_gene[:5]}"
        )

    prototypes = torch.stack(rows, dim=0).detach()
    prototypes.requires_grad = False
    return prototypes, kinds
