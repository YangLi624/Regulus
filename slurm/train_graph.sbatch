#!/usr/bin/env bash
#SBATCH -J graph_train
#SBATCH -o outputs/logs/train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=1

set -euo pipefail

mkdir -p outputs/logs

CONFIG="${1:-${CONFIG:-configs/graph_config.yaml}}"
SUFFIX="${SUFFIX:-graph_train}"
GRAPH_ASSET_DIR="${GRAPH_ASSET_DIR:-}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
nvidia-smi

ARGS=(
  --config "${CONFIG}"
  --output-suffix "${SUFFIX}"
  --gpu 0
)
if [[ -n "${GRAPH_ASSET_DIR}" ]]; then
  ARGS+=(--graph-asset-dir "${GRAPH_ASSET_DIR}")
fi

regulus graph-train "${ARGS[@]}"

echo "regulus graph-train completed."
