# Versioned graph assets

Graph training and frozen HGT inference consume one versioned asset root instead
of independent file paths. The default release is
`assets/graph/regulus_graph_v1/`, configured by `data.graph_asset_dir` in
`configs/graph_config.yaml`.

## Stable runtime layout

- `manifest.json`: schema, file roles, shapes, sizes, and SHA-256 hashes.
- `metadata.json`: node counts and feature-construction summary.
- `edges/`: the five canonical heterogeneous relations.
- `embeddings/`: six model inputs plus the stable gene order.
- `nodes/`: TF, Gene, Cell type, and CFO index tables.
- `universes/`: stable Gene and TF universes.
- `cache/hetero_data.pt`: optional PyG cache; never the sole archival representation.

Large files are deposited outside Git. After extraction, validate every recorded
file before training:

```bash
python knowledge/graph_build/create_asset_manifest.py \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --check
```

`regulus graph-train` performs the same validation by default. To use an asset in
another location:

```bash
regulus graph-train \
  --config configs/graph_config.yaml \
  --graph-asset-dir /path/to/regulus_graph_v1
```

The `data.processed_dir` configuration key and `--data-dir` CLI spelling are
accepted only as migration aliases. New configs and documentation must use
`graph_asset_dir`.

## Rebuilding

`knowledge/graph_build/run_preprocessing.py` materializes the stable runtime files
and writes a new manifest. Local PCA and normalization intermediates are kept in
`data/graph_build_intermediate/` and are not published. The Gene and TF universes,
Geneformer embeddings, and gene order are precomputed build inputs and are
explicitly identified in the manifest. A rebuilt graph must receive a new asset
ID or version whenever any of these inputs, node orders, edge tables, or
feature-generation models change.
