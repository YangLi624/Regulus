#!/usr/bin/env bash
set -euo pipefail

regulus manipulate \
  --bundle-path bundles/joung-tfatlas-cfo-post-state-v1 \
  --anchor-h5ad outputs/example_cells.regulus.h5ad \
  --cfo-targets GO:0000723 \
  --cfo-delta 0.2 \
  --mode cfo_only \
  -o outputs/manipulate
