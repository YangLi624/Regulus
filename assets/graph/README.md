# Regulus graph assets

`regulus_graph_v1/` is the local layout for the versioned graph release. Large
CSV, NPY, and PT files are distributed through the external data deposit and are
intentionally excluded from Git. Local preprocessing intermediates are not part
of the public graph asset.

The tracked `manifest.json` records the graph schema, required files, shapes,
sizes, SHA-256 hashes, source commit, and authoritative paper-checkpoint hash.
After downloading and extracting the asset, validate it by running:

```bash
python knowledge/graph_build/create_asset_manifest.py \
  --graph-asset-dir assets/graph/regulus_graph_v1 \
  --check
```

Graph training uses `configs/graph_config.yaml`. A different extraction location
can be supplied with `regulus graph-train --graph-asset-dir <path>`.
