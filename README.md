# Regulus

Regulus learns a heterogeneous functional-regulatory representation spanning
transcription factors (TFs), genes, cell types, and cellular function ontology
(CFO) terms. This first public release contains the graph-training component.
Perturbation prediction, explanation, and virtual functional manipulation will
be added after their release interfaces are finalized.

## Graph schema

| Source | Relation | Target | Evidence |
| --- | --- | --- | --- |
| TF | `regulates` | Gene | TF-target regulatory evidence |
| Gene | `annotated_to` | CFO | Gene-to-function annotation |
| Cell type | `scenic_activity` | TF | SCENIC AUCell regulon activity |
| Cell type | `llm_context` | CFO | LLM-assisted cell-context association |
| TF | `llm_regulates` | CFO | LLM-assisted continuous TF-CFO confidence |

The TF-CFO relation contributes to HGT message passing but is not directly
reconstructed or used as a cross-type alignment target.

## Installation

Python 3.9 or newer is required. Install a PyTorch build suitable for the local
CUDA environment, then install Regulus:

```bash
pip install -e .
regulus graph-train --help
```

## Graph assets

Large graph files and the paper checkpoint are distributed separately through
Zenodo. Their DOI will be added to this README when the public deposit is
released. The repository tracks a manifest containing the expected file names,
sizes, shapes, and SHA-256 hashes.

Extract the asset to `assets/graph/regulus_graph_v1/`, then validate it:

```bash
python -c "from regulus.graph.assets import validate_graph_asset; validate_graph_asset('assets/graph/regulus_graph_v1')"
```

The public asset contains 19 required runtime files under `edges/`,
`embeddings/`, `nodes/`, and `universes/`. Local preprocessing models and
construction-time audit files are not part of the release.

## Training

```bash
regulus graph-train \
  --config configs/graph_config.yaml \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --output-dir outputs \
  --output-suffix graph_v1
```

For SLURM:

```bash
sbatch slurm/train_graph.sbatch configs/graph_config.yaml
```

The paper checkpoint uses the internal node key `go` for CFO nodes. This key is
retained as a serialization contract while public documentation uses CFO.

## Repository scope

- `regulus/graph/`: graph construction from released assets, HGT model, losses,
  training, checkpoint loading, and asset validation.
- `configs/graph_config.yaml`: graph-training configuration.
- `assets/graph/regulus_graph_v1/manifest.json`: public asset contract.
- `docs/user-guide/graph-assets.md`: asset layout and validation details.
- `tests/`: lightweight schema and release-metadata checks.

## License and citation

Regulus is released under the MIT License. Citation metadata are provided in
`CITATION.cff`.
