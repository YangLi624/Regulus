#!/usr/bin/env bash
#SBATCH -J train_perturb
#SBATCH -o outputs/logs/train_perturb_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1

# Submit perturbation-model training through the public CLI.
# Usage: sbatch slurm/train_perturb.sbatch configs/perturb_config.yaml

set -euo pipefail

mkdir -p outputs/logs

CONFIG="${1:-${CONFIG:-configs/perturb_config.yaml}}"
SEED="${SEED:-42}"

nvidia-smi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m regulus.cli.main perturb-train \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --gpu 0
