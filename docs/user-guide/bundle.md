# Model bundles

A released bundle is a self-contained inference unit. Every manifest path is
relative to the bundle root and may not escape that directory.

```text
bundle/
  manifest.json
  config/perturb_config.yaml
  checkpoints/graph_model.pt
  checkpoints/perturb_model.pt
  graph/manifest.json
  graph/... required graph assets
  graph/universes/gene_universe.csv
```

The manifest declares input representation, trained channels, head type,
supported runtime modes, graph asset, exact checkpoints, and gene universe.
Optional `files` entries record file sizes and SHA-256 checksums.

```json
{
  "schema_version": "1.0",
  "bundle_id": "norman-post-state-v1",
  "representation": "post_state",
  "channels": "gene_cfo",
  "head": "prototype_matching",
  "supported_modes": ["joint", "gene_only", "cfo_only"],
  "default_mode": "joint",
  "train_config": "config/perturb_config.yaml",
  "graph_ckpt": "checkpoints/graph_model.pt",
  "perturb_ckpt": "checkpoints/perturb_model.pt",
  "graph_asset": "graph/manifest.json",
  "gene_universe": "graph/universes/gene_universe.csv"
}
```

The perturbation checkpoint contains candidate names, candidate types, and
their order. Loading a bundle never searches the source repository for missing
files.
