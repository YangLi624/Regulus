"""Optional compact visualization of Regulus evidence paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def _read_row(path: Path, cell_id: Optional[str]) -> Optional[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return None
    if cell_id is None:
        return rows[0]
    return next((row for row in rows if str(row.get("cell_id")) == str(cell_id)), None)


def render_evidence_network_pdfs(
    evidence_jsonl: str | Path,
    output_dir: str | Path,
    *,
    cell_id: Optional[str] = None,
) -> list[Path]:
    """Render one sample's raw evidence paths to a lightweight PDF."""
    row = _read_row(Path(evidence_jsonl), cell_id)
    if row is None:
        return []
    graph = nx.DiGraph()
    for path in row.get("paths", []):
        tf = str(path.get("tf", ""))
        gene = str(path.get("gene", ""))
        cfo = str(path.get("cfo", "") or "")
        source = str(path.get("evidence_source", path.get("evidence_level", "reference_supported")))
        attributes = {"source": source}
        if tf and gene:
            graph.add_edge(tf, gene, **attributes)
        if gene and cfo:
            graph.add_edge(gene, cfo, **attributes)
        elif tf and cfo:
            graph.add_edge(tf, cfo, **attributes)
    if not graph:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{row.get('cell_id', 'sample')}_evidence.pdf"
    figure, axis = plt.subplots(figsize=(8, 6))
    positions = nx.spring_layout(graph, seed=42)
    node_colors = []
    for node in graph.nodes:
        if str(node).startswith("GO:"):
            node_colors.append("#4477AA")
        elif graph.out_degree(node):
            node_colors.append("#66AA55")
        else:
            node_colors.append("#CC6677")
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=650, ax=axis)
    reference_edges = [edge for edge in graph.edges if graph.edges[edge]["source"] != "llm_context"]
    llm_edges = [edge for edge in graph.edges if graph.edges[edge]["source"] == "llm_context"]
    nx.draw_networkx_edges(graph, positions, edgelist=reference_edges, width=1.2, ax=axis)
    nx.draw_networkx_edges(graph, positions, edgelist=llm_edges, width=1.2, style="dashed", ax=axis)
    nx.draw_networkx_labels(graph, positions, font_size=7, ax=axis)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return [output]
