# Perturbation H5AD format

## Matrices

- `adata.X`: gene matrix aligned by `var_names` to the bundle gene universe.
- `adata.obsm['X_cfo_activity']`: CFO activity matrix for CFO-channel models.
- `adata.uns['regulus_cfo_ids']`: ordered CFO identifiers for the matrix columns.

`representation: delta` and `representation: post_state` describe matrices
prepared before they enter Regulus. Regulus does not convert between them.

## Metadata

Training data require `obs['perturbation']`; prediction data do not require
labels or control cells. If prediction data contain rows labelled `control`,
Regulus retains them; only supervised training excludes those rows. CFO
identifiers and their order must match the bundle graph asset. Use
`regulus preprocess --bundle-path BUNDLE` to compute them.
