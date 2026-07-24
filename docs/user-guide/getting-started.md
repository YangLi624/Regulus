# Getting started

This guide installs Regulus, downloads a pretrained model bundle, computes CFO
activity, and produces a first candidate-driver ranking.

## Installation

Regulus requires Python 3.10 or newer. We recommend a dedicated Conda
environment:

```bash
conda create -n regulus python=3.10 -y
conda activate regulus
python -m pip install --upgrade pip
```

### PyTorch and accelerator support

Regulus depends on PyTorch and PyTorch Geometric. The default installation can
run on CPU, but a CUDA-capable GPU is recommended for training and larger
inference jobs. GPU users should first install the PyTorch build matching their
operating system, Python version, and CUDA runtime using the
[official PyTorch selector](https://pytorch.org/get-started/locally/).

After PyTorch is available, install Regulus from the source repository:

```bash
git clone https://github.com/YangLi624/Regulus.git
cd Regulus
python -m pip install .
```

Confirm that the command-line interface and package import are available:

```bash
regulus --help
python -c "import regulus; print(regulus.__version__)"
```

For editable development and tests:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

To rebuild graph inputs, install the knowledge-building dependencies:

```bash
python -m pip install -e ".[knowledge]"
```

## Download a model

The Python package contains the CLI and model code. Pretrained weights and their
matching graph assets are hosted separately as versioned bundles.

List the published bundles:

```bash
regulus download
```

Download a bundle:

```bash
regulus download norman-post-state-v1 --output-dir bundles
```

Regulus downloads the archive, verifies its SHA-256 checksum, and installs it
under:

```text
bundles/norman-post-state-v1/
```

Passing `--force` replaces an existing installation. A bundle is
self-contained; inference never searches the source repository for missing
checkpoints or graph files.

## Prepare the input

Start from an H5AD file whose `adata.var_names` contain gene symbols and whose
`adata.X` contains the expression values you want the model to consume. Pass
`--layer` only when CFO activity should be scored from a different expression
layer; this option does not replace `adata.X`.

Released gene+CFO and CFO-only bundles require CFO activity:

```bash
regulus preprocess \
  -i examples/data/example_cells.h5ad \
  -o outputs/cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1
```

To score a layer instead of `adata.X`:

```bash
regulus preprocess \
  -i examples/data/example_cells.h5ad \
  -o outputs/cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --layer log1p
```

The command uses the CFO-gene membership encoded in the selected graph asset,
computes UCell scores, restores the original `adata.X`, and writes a new H5AD.
It adds:

- `adata.obsm["X_cfo_activity"]`;
- `adata.uns["regulus_cfo_ids"]`;
- `adata.uns["regulus_cfo_activity_method"]`.

Existing outputs are not overwritten unless `--overwrite` is provided.
The repository example is synthetic and intended only to validate the software
interface; it is not a benchmark dataset.

## Run a first prediction

```bash
regulus predict \
  -i outputs/cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --top-k 50 \
  -o outputs/predictions.csv
```

The output contains one ranked candidate list per input cell. `--mode` defaults
to the bundle's declared mode; setting it explicitly is useful when comparing
the `joint`, `gene_only`, and `cfo_only` views of a compatible gene+CFO model.

## Next steps

- Read [Input data and model bundles](input-data-and-bundles.md) before using a
  new dataset or choosing between `post_state` and `delta`.
- Read [Advanced usage](advanced-usage.md) for attribution, evidence paths,
  virtual CFO manipulation, training, and graph-asset maintenance.
- Runnable command examples are available in `examples/`.

## Common installation checks

If `regulus` is not found, verify that the environment used for installation is
active and run:

```bash
python -m pip show regulus
python -m regulus.cli.main --help
```

If a GPU is not detected, check the PyTorch installation independently:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If preprocessing reports that `pyucell` is unavailable, reinstall Regulus in
the active environment:

```bash
python -m pip install .
```
