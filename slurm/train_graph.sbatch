#!/bin/bash
#SBATCH -J graph_train
#SBATCH -o outputs/logs/train_%j.out
#SBATCH -p a100
#SBATCH --nodelist=n06
#SBATCH --qos=low
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1

mkdir -p outputs/logs
source activate regulus

CONFIG="${1:-${CONFIG:-configs/graph_config.yaml}}"
SUFFIX="${SUFFIX:-graph_train}"
GPU="${GPU:-0}"
GRAPH_ASSET_DIR="${GRAPH_ASSET_DIR:-}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"

ARGS=(
  --config "${CONFIG}"
  --output-suffix "${SUFFIX}"
  --gpu "${GPU}"
)
if [[ -n "${GRAPH_ASSET_DIR}" ]]; then
  ARGS+=(--graph-asset-dir "${GRAPH_ASSET_DIR}")
fi

regulus graph-train "${ARGS[@]}"

echo "regulus graph-train completed."
