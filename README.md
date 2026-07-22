# Regulus

Regulus links single-cell perturbation states to candidate transcriptional
regulators through a frozen heterogeneous graph and a compact perturbation
model. The released code covers graph training, perturbation training,
prediction, attribution, and virtual CFO manipulation.

## Installation

```bash
pip install .
regulus --help
```

## Graph model

The graph is trained from the versioned asset directory declared in
`configs/graph_config.yaml`:

```bash
regulus graph-train --config configs/graph_config.yaml --output-suffix graph
```

The public graph asset contains node universes, node features, typed edges and
a manifest. Intermediate preprocessing files are not required at runtime.

## Perturbation model

The public model surface has three independent choices:

| Field | Values | Meaning |
|---|---|---|
| `representation` | `delta`, `post_state` | Semantic meaning of the supplied matrix |
| `channels` | `gene`, `cfo`, `gene_cfo` | Channels used to construct the trained model |
| `head` | `mlp`, `prototype_matching` | Independent perturbation classifier |

Regulus consumes the supplied matrices as-is. It does not infer control cells,
calculate a control mean, derive fold changes, or substitute a zero baseline.
A `delta` matrix must therefore be prepared before training or prediction.

`channels: gene` routes to `GeneTokenTransformerEncoder`. `channels: cfo` and
`channels: gene_cfo` route to `JointCrossTransformerEncoder`. A trained
gene+CFO model supports runtime modes `joint`, `gene_only`, and `cfo_only`;
these modes never replace the trained encoder.

Train with:

```bash
regulus perturb-train --config configs/perturb_config.yaml
```

The release includes four pretrained-model recipes:

- `configs/perturb_norman_post_state.yaml`
- `configs/perturb_schmidt_post_state.yaml`
- `configs/perturb_tian_crispra_post_state.yaml`
- `configs/perturb_joung_cfo_post_state.yaml`

## H5AD input

Gene values are read from `adata.X` and aligned to `gene_universe.csv`.
CFO-channel models additionally require `adata.obsm['X_cfo_activity']`.
CFO identifiers are stored in `adata.uns['regulus_cfo_ids']`. Training files
require `obs['perturbation']`; prediction files do not. Rows labelled `control` are
excluded from supervised training and retained during prediction, explanation,
and anchor construction. They are never used to transform another input row.

## Prediction

```bash
regulus predict \
  -i prepared_cells.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --top-k 50 \
  -o predictions.csv
```

The bundle stores the exact graph checkpoint, perturbation checkpoint,
candidate order, graph asset, training configuration and input semantics needed
to reconstruct the model without the training H5AD.

## Explanation

```bash
regulus explain \
  -i prepared_cells.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --attribution-channel gene \
  --attribution-method gradient_x_input \
  --obs-filter "cell_type == 'T cell'" \
  -o explain_out
```

Outputs are `scores.csv`, long-form `attributions.csv`,
`evidence_paths.jsonl`, and `explain_manifest.json`. By default both the
attribution target and graph context use the top predicted driver. Increase
`--top-k-candidates` when several predicted drivers should constrain the graph.

## Virtual CFO manipulation

```bash
regulus manipulate \
  --bundle-path bundles/joung-tfatlas-cfo-post-state-v1 \
  --anchor-h5ad prepared_cells.h5ad \
  --cfo-targets "peptide hormone secretion" \
  --cfo-delta 0.5 \
  --mode cfo_only \
  --sort-by rank_gain \
  -o manipulate_out
```

Manipulation adds the requested value to the selected CFO coordinate of an
anchor profile and reports candidate rank and score changes.

## Repository layout

| Path | Content |
|---|---|
| `regulus/graph/` | Graph schema, model and training code |
| `regulus/perturb/` | Unified perturbation model, data and training code |
| `regulus/explain/` | Attribution and evidence extraction |
| `regulus/manipulate/` | Virtual CFO edit orchestration and target resolution |
| `assets/graph/` | Versioned graph assets |
| `configs/` | Graph and perturbation recipes |
| `regulus/assets/bundles/` | Download catalog for pretrained bundles |
| `tests/` | Unit and contract tests |

## Tests

```bash
pytest tests -q
```
