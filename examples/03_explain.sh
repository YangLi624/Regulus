#!/usr/bin/env bash
set -euo pipefail

regulus explain \
  -i outputs/example_cells.regulus.h5ad \
  --bundle-path bundles/norman-post-state-v1 \
  --mode joint \
  --attribution-channel gene \
  --attribution-method gradient_x_input \
  --top-k-candidates 1 \
  -o outputs/explain
