# CLI reference

```bash
regulus graph-train --config configs/graph_config.yaml
regulus perturb-train --config configs/perturb_config.yaml
regulus predict -i cells.h5ad --bundle-path BUNDLE --mode joint -o predictions.csv
regulus explain -i cells.h5ad --bundle-path BUNDLE --mode joint -o explain_out
regulus manipulate --bundle-path BUNDLE --anchor-h5ad cells.h5ad \
  --cfo-targets "GO:0000001" --cfo-delta 0.5 -o manipulate_out
```

## Predict

- `--input-representation`: optional `delta` or `post_state` bundle consistency check.
- `--mode`: `gene_only`, `cfo_only`, or `joint`; defaults to the bundle mode.
- `--top-k`: number of candidates returned per cell.

Regulus does not derive a baseline or transform the supplied representation.

## Explain

- `--attribution-channel`: `gene`, `cfo`, or `both`.
- `--attribution-method`: `gradient_x_input` or `integrated_gradients`.
- `--target`: optional explicit candidate; the default is the top prediction.
- `--top-k-candidates`: predicted drivers used to constrain graph paths; default 1.
- `--plot`: optionally render one compact network PDF.

Graph paths are centered on the predicted driver and attributed features.
`reference_supported` denotes non-LLM graph relations and `llm_context`
denotes TF-CFO context retained only with a gene mediator.

## Manipulate

`--mode cfo_only` uses the CFO stream and `--mode joint` uses both streams of
a gene+CFO checkpoint. `--sort-by` accepts `rank_gain`, `delta`, or `after`.
