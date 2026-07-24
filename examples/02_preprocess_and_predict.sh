#!/usr/bin/env bash
set -euo pipefail

regulus preprocess \
  -i examples/data/example_cells.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  -o outputs/example_cells.regulus.h5ad

regulus predict \
  -i outputs/example_cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  -o outputs/predictions.csv
