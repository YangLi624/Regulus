# Input data and model bundles

Regulus aligns every input to the exact feature order and model semantics stored
in a released bundle. This page defines that contract.

## H5AD matrices

### Gene expression

- Gene values are read from `adata.X`.
- Gene symbols are read from `adata.var_names`.
- At model load time, genes are aligned to the bundle's
  `gene_universe.csv`; genes absent from the input receive zero values.
- Use `regulus preprocess --layer NAME` when CFO activity should be calculated
  from an expression layer rather than `adata.X`.

Regulus does not normalize, log-transform, or batch-correct the gene matrix.
Supply values whose preprocessing is appropriate for the selected model and
analysis.

### CFO activity

Models trained with `channels: cfo` or `channels: gene_cfo` require:

- `adata.obsm["X_cfo_activity"]`: cells by CFOs;
- `adata.uns["regulus_cfo_ids"]`: CFO identifiers in column order.

Run `regulus preprocess` with the same bundle that will be used for inference.
The command computes UCell scores from CFO member genes and records the exact CFO
order. Regulus rejects a CFO matrix whose number or order of columns does not
match the graph asset.

## Input representation

The representation is fixed when a model is trained and recorded in its bundle.
`--input-representation` is an optional consistency check; it does not convert
one representation into another.

### Post-state

With `representation: post_state`, Regulus consumes the supplied gene and CFO
matrices as-is. Prediction files do not require perturbation labels or control
cells.

### Delta

With `representation: delta`, the H5AD must contain:

```text
adata.obs["perturbation"]
```

with at least one value equal to `control`. Regulus calculates the mean of the
control rows and subtracts it from every gene row. CFO-channel models apply the
same operation to `X_cfo_activity`.

Missing controls raise an error. Regulus does not infer alternative control
labels and never substitutes a zero baseline. During supervised training,
control rows define the reference and are then excluded from candidate-label
training. Prediction and explanation retain all requested rows.

## Training metadata

Perturbation training requires:

- `obs["perturbation"]`: the target candidate label or `control`;
- genes in `var_names`;
- CFO activity and identifiers when the selected channels include CFO.

Candidate names and their order are saved in the perturbation checkpoint and
reused at inference. A score is therefore comparable only within the candidate
space declared by that checkpoint.

## Channels and runtime modes

The trained channel architecture is fixed:

| `channels` | Encoder | Available information |
|---|---|---|
| `gene` | `GeneTokenTransformerEncoder` | Gene expression only |
| `cfo` | `JointCrossTransformerEncoder` | CFO activity only |
| `gene_cfo` | `JointCrossTransformerEncoder` | Gene and CFO streams |

A gene+CFO checkpoint can expose `joint`, `gene_only`, and `cfo_only` runtime
modes. These modes mask or combine streams within the trained architecture; they
do not replace the encoder or create a separately trained model. Always use a
mode listed in the bundle manifest.

## Released bundles

| Bundle ID | Representation | Channels | Intended use |
|---|---|---|---|
| `norman-post-state-v1` | `post_state` | `gene_cfo` | Broad single-gene CRISPRa inference |
| `schmidt-post-state-v1` | `post_state` | `gene_cfo` | CRISPRa inference in an independent context |
| `tian-crispra-post-state-v1` | `post_state` | `gene_cfo` | CRISPRa prediction and attribution workflows |
| `joung-tfatlas-cfo-post-state-v1` | `post_state` | `cfo` | CFO-centered inference and virtual manipulation |

Use `regulus download` to read the catalog embedded in the installed package.
The catalog records the archive URL, size, and SHA-256 checksum.

## Bundle contents

A released bundle is a self-contained inference unit:

```text
bundle/
  manifest.json
  config/perturb_config.yaml
  checkpoints/graph_model.pt
  checkpoints/perturb_model.pt
  graph/manifest.json
  graph/edges/
  graph/embeddings/
  graph/nodes/
  graph/universes/gene_universe.csv
```

The root `manifest.json` declares:

- bundle and schema versions;
- `representation`, trained `channels`, and head type;
- supported and default runtime modes;
- graph and perturbation checkpoints;
- graph-asset and gene-universe paths;
- candidate and input semantics;
- optional file sizes and SHA-256 checksums.

All manifest paths are relative to the bundle root and may not escape it.
Loading a bundle never searches the source checkout, a training H5AD, or an
unversioned data directory for missing files.

## Selecting a bundle

Choose a bundle by the biological perturbation setting, candidate space, input
representation, and channels that match the intended analysis. Released models
are useful starting points, not universal causal models. Report the exact bundle
ID, runtime mode, input preprocessing, and any cell filtering when sharing
results.
