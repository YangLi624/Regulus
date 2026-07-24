# Advanced usage

This page covers interpretation, virtual function manipulation, model training,
and graph-asset maintenance. A released bundle is sufficient for prediction,
explanation, and manipulation; graph training is not part of the standard user
workflow.

## Explain predictions

```bash
regulus explain \
  -i outputs/cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --attribution-channel gene \
  --attribution-method gradient_x_input \
  --top-k-features 20 \
  --top-k-candidates 1 \
  -o outputs/explain
```

Available attribution channels are `gene`, `cfo`, and `both`. Available methods
are `gradient_x_input` and `integrated_gradients`.

By default, attribution targets the top predicted driver and graph paths are
constructed around that same driver. `--target` selects an explicit candidate.
`--top-k-candidates` allows several top-ranked candidates to constrain evidence
path construction. The attributed features remain target-specific to the
selected attribution target.

Use `--obs-filter` with a Pandas query to select cells before explanation:

```bash
regulus explain \
  -i outputs/cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --obs-filter "cell_type == 'T cell'" \
  --plot \
  -o outputs/explain_t_cells
```

The output directory contains:

- `scores.csv`: the full candidate score matrix;
- `attributions.csv`: long-form signed feature attributions;
- `evidence_paths.jsonl`: predicted-driver-centered evidence paths;
- `explain_manifest.json`: bundle, method, parameters, and output metadata;
- `network_viz/`: optional compact network PDFs when `--plot` is used.

Evidence sources are reported conservatively:

- `reference_supported`: non-LLM regulatory or annotation relationships;
- `llm_context`: literature-supported TF-CFO context, retained in a path only
  when a gene mediator is present.

These paths organize model-associated evidence. They should not be described as
experimentally validated regulatory mechanisms.

## Virtual CFO manipulation

```bash
regulus manipulate \
  --bundle-path bundles/joung-tfatlas-cfo-post-state-v1 \
  --anchor-h5ad outputs/cells.regulus.h5ad \
  --cfo-targets "peptide hormone secretion" \
  --cfo-delta 0.5 \
  --mode cfo_only \
  --sort-by rank_gain \
  --top-k 20 \
  -o outputs/manipulate
```

Targets may be CFO identifiers or names. Multiple targets and deltas are
comma-separated. The current edit mode, `anchor_plus_delta`, adds each requested
value to the selected coordinate of the mean anchor profile.

`manipulation_rankings.csv` reports:

- candidate rank and score before editing;
- candidate rank and score after editing;
- rank gain and score change.

`--sort-by` accepts `rank_gain`, `delta`, or `after`. Use `--obs-filter` to
define the anchor cohort. Functional editing is a model-based prioritization
operation and does not imply that an experimental intervention will produce the
same magnitude or direction of effect.

## Train a perturbation model

Perturbation training is controlled by a portable YAML configuration:

```bash
regulus perturb-train --config configs/perturb_config.yaml
```

The stable public choices are:

| Field | Values |
|---|---|
| `representation` | `delta`, `post_state` |
| `channels` | `gene`, `cfo`, `gene_cfo` |
| `head` | `mlp`, `prototype_matching` |

`channels: gene` uses `GeneTokenTransformerEncoder`; `channels: cfo` and
`channels: gene_cfo` use `JointCrossTransformerEncoder`. Prototype matching
compares the encoded state against candidate TF or gene prototypes from the
frozen graph checkpoint. The MLP head is retained as an independent baseline.

The repository includes example release recipes:

- `configs/perturb_norman_post_state.yaml`;
- `configs/perturb_schmidt_post_state.yaml`;
- `configs/perturb_tian_crispra_post_state.yaml`;
- `configs/perturb_joung_cfo_post_state.yaml`.

These are training templates. Update dataset paths and output locations before
use. See [Input data and model bundles](input-data-and-bundles.md) for control
handling and H5AD requirements.

## Build a model bundle

After training, package the exact perturbation checkpoint together with its
graph checkpoint and graph asset:

```bash
regulus bundle-build \
  --bundle-id my-model-v1 \
  --config configs/my_perturb_config.yaml \
  --perturb-checkpoint outputs/perturb/best_model.pt \
  --graph-checkpoint outputs/graph/best_model.pt \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --output-dir dist/bundles
```

The resulting directory is relocatable and records all paths relative to the
bundle root.

## Train the heterogeneous graph

Graph training is an advanced developer workflow. It is unnecessary when using
a released model bundle.

The graph trainer consumes one versioned asset root declared by
`data.graph_asset_dir` in `configs/graph_config.yaml`:

```bash
regulus graph-train \
  --config configs/graph_config.yaml \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --output-dir outputs \
  --output-suffix graph
```

The stable graph asset contains node tables, typed edge tables, model input
embeddings, universes, and a manifest. Intermediate preprocessing files are not
required at training or inference time.

### Released graph scale

`regulus-graph-v1` uses one frozen, manifest-validated node and edge universe:

| Node type | Count |
|---|---:|
| Transcription factors | 925 |
| Genes | 20,094 |
| Cell types | 303 |
| CFO terms | 613 |
| **Total nodes** | **21,935** |

| Relation | Count |
|---|---:|
| TF-gene reference-supported regulation | 642,731 |
| Gene-CFO annotation support | 50,652 |
| Cell type-TF SCENIC activity | 280,275 |
| Cell type-CFO LLM context | 3,949 |
| TF-CFO LLM context | 6,301 |
| **Total edges** | **983,908** |

These counts are defined by `assets/graph/regulus_graph_v1/manifest.json`.
Changing a node universe, node order, or edge table creates a new graph asset
version rather than silently changing `regulus-graph-v1`.

### Graph-asset layout

```text
assets/graph/regulus_graph_v1/
  manifest.json
  metadata.json
  edges/
  embeddings/
  nodes/
  universes/
  cache/hetero_data.pt
```

- `manifest.json` records the schema, file roles, shapes, sizes, and SHA-256
  hashes.
- `metadata.json` records node counts and feature-construction summaries.
- `edges/` stores the canonical heterogeneous relations.
- `embeddings/` stores graph-model inputs and stable feature order.
- `nodes/` stores TF, gene, cell-type, and CFO index tables.
- `universes/` stores the stable gene and TF universes.
- `cache/hetero_data.pt` is an optional PyTorch Geometric cache, never the sole
  archival representation.

Validate an extracted asset before training:

```bash
python knowledge/graph_build/create_asset_manifest.py \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --check
```

`regulus graph-train` performs the same validation by default. The deprecated
`data.processed_dir` key and `--data-dir` spelling are accepted only as migration
aliases; new configurations should use `graph_asset_dir`.

### Rebuilding a graph asset

Install the optional knowledge dependencies, then run:

```bash
python knowledge/graph_build/run_preprocessing.py
```

The preprocessing workflow materializes the stable runtime files and writes a
new manifest. Local PCA and normalization intermediates remain build artifacts
and are not part of the public runtime asset. Assign a new asset ID or version
whenever node order, edge tables, universes, source embeddings, or
feature-generation models change.

## CLI discovery

Every command provides focused help:

```bash
regulus download --help
regulus preprocess --help
regulus predict --help
regulus explain --help
regulus manipulate --help
regulus perturb-train --help
regulus graph-train --help
regulus bundle-build --help
```
