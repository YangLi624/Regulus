# Regulus

Regulus is a function-centered framework for inverse driver inference from
single-cell transcriptomic states. Instead of treating cellular function only
as a post hoc enrichment result, Regulus represents gene expression, cellular
function, candidate regulators, genes, and cellular contexts in a shared
functional-regulatory manifold.

Given an observed cellular state, Regulus can:

- rank candidate transcription factor and gene drivers;
- attribute predictions to gene and cellular-function inputs;
- trace attributed features through regulator-gene-function evidence paths;
- virtually edit selected cellular functions and quantify candidate rank shifts.

The functional channel uses a compact cellular function ontology (CFO) derived
from context-aware Gene Ontology terms. Its graph-pretrained representations
combine reference-supported regulatory and annotation relationships,
single-cell context priors, and literature-supported LLM context. Regulus
outputs ranked hypotheses and traceable evidence; experimental validation is
still required to establish causal regulation.

## Quick start

### 1. Install Regulus

Regulus requires Python 3.10 or newer. A fresh environment is recommended:

```bash
conda create -n regulus python=3.10 -y
conda activate regulus

git clone https://github.com/YangLi624/Regulus.git
cd Regulus
python -m pip install --upgrade pip
python -m pip install .

regulus --help
```

GPU users should install a PyTorch build compatible with their CUDA environment
before `python -m pip install .`. See the
[installation guide](docs/user-guide/getting-started.md#installation) for CPU,
GPU, and development setup.

### 2. Download a pretrained bundle

Model weights and their matching graph assets are distributed as self-contained
bundles rather than stored in the Python package:

```bash
# List available bundles.
regulus download

# Download and verify one bundle.
regulus download norman-post-state-v1 --output-dir bundles
```

Each download is checked against its published SHA-256 hash before extraction.
The bundle contains the graph checkpoint, perturbation checkpoint, exact
candidate order, graph asset, input semantics, and training configuration
needed for inference.

### 3. Prepare an H5AD file

Regulus reads model gene inputs from `adata.X` using `adata.var_names`. Models
with a CFO channel additionally require CFO activity scores, which may be
computed from `adata.X` or a selected expression layer. Compute them with the
matching bundle:

```bash
regulus preprocess \
  -i examples/data/example_cells.h5ad \
  -o outputs/example_cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1
```

Preprocessing preserves the gene matrix and adds:

- `adata.obsm["X_cfo_activity"]`: UCell CFO activity scores;
- `adata.uns["regulus_cfo_ids"]`: CFO identifiers in matrix-column order.

### 4. Rank candidate drivers

```bash
regulus predict \
  -i outputs/example_cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --top-k 50 \
  -o outputs/predictions.csv
```

The included H5AD is synthetic and intended only for interface validation.
`outputs/predictions.csv` contains the top-ranked candidate drivers and model
scores for each input cell. A gene+CFO bundle supports `joint`, `gene_only`, and
`cfo_only` runtime modes; a CFO-only bundle supports only its declared modes.

## Choose a bundle

| Bundle | Trained input | Dataset background |
|---|---|---|
| `norman-post-state-v1` | Gene + CFO, post-state | Norman et al. Perturb-seq CRISPRa profiles from K562 cells; the bundle uses the single-gene subset |
| `schmidt-post-state-v1` | Gene + CFO, post-state | Schmidt et al. CRISPRa profiles from primary human T cells under resting and restimulated conditions |
| `tian-crispra-post-state-v1` | Gene + CFO, post-state | Tian et al. CRISPRa profiles from human iPSC-derived neurons |
| `joung-tfatlas-cfo-post-state-v1` | CFO only, post-state | Joung et al. TFAtlas profiles from transcription-factor overexpression during directed differentiation of human pluripotent stem cells |

Released bundles are designed for direct use and method exploration. They are
not article-reproduction packages or substitutes for dataset-specific
validation.

## Input representations

`post_state` bundles consume the supplied gene and CFO matrices as-is and do
not require control cells.

For a `delta` model, the input H5AD must contain rows where
`obs["perturbation"] == "control"`. Regulus calculates the control mean and
subtracts it from every gene row and, when present, every CFO row. Missing
controls raise an error; Regulus never substitutes a zero baseline.

See [Input data and model bundles](docs/user-guide/input-data-and-bundles.md)
for the complete H5AD and bundle contracts.

## Explain a prediction

```bash
regulus explain \
  -i outputs/example_cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --attribution-channel gene \
  --attribution-method gradient_x_input \
  -o explain_out
```

The default explanation is centered on the top predicted driver. Outputs
include prediction scores, long-form feature attributions, graph-constrained
evidence paths, and a run manifest. Integrated gradients, CFO attribution,
multiple candidate drivers, cell filtering, and an optional network plot are
also available.

## Manipulate a cellular function

```bash
regulus manipulate \
  --bundle-path bundles/joung-tfatlas-cfo-post-state-v1 \
  --anchor-h5ad outputs/example_cells.regulus.h5ad \
  --cfo-targets "peptide hormone secretion" \
  --cfo-delta 0.5 \
  --mode cfo_only \
  --sort-by rank_gain \
  -o manipulate_out
```

This command adds the requested value to selected CFO coordinates of an anchor
state and reports the resulting candidate score and rank changes. It is an in
silico prioritization tool, not a direct prediction of experimental effect
size.

## Documentation

- [Getting started](docs/user-guide/getting-started.md): installation,
  downloads, preprocessing, and a first prediction.
- [Input data and model bundles](docs/user-guide/input-data-and-bundles.md):
  H5AD requirements, representations, model modes, and bundle contents.
- [Explanation, manipulation, and training](docs/user-guide/advanced-usage.md):
  attribution, virtual CFO editing, perturbation-model training, bundle
  construction, graph training, and asset validation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Please cite Regulus using the metadata in
[`CITATION.cff`](CITATION.cff); the v1.0.0 source archive is available on
[Zenodo](https://doi.org/10.5281/zenodo.21522549). Model bundles, the frozen
graph, and their checksums are archived separately on
[Zenodo](https://doi.org/10.5281/zenodo.21488583). Regulus is released under
the [MIT License](LICENSE).
