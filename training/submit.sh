#!/bin/bash
# Generic SLURM job script for RF-DETR Keypoint training.
# Adjust the variables below for your cluster before submitting.

# ── Cluster-specific settings ────────────────────────────────────────────────
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4        # one task per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu            # replace with your cluster's GPU partition
#SBATCH --job-name=rfdetr_dog_pose
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# ─────────────────────────────────────────────────────────────────────────────

# Path to the conda/venv environment that has rfdetr installed
ENV_PATH="${HOME}/envs/animal2robot"

# Root of the repository (defaults to the directory containing this script)
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p "${REPO_DIR}/logs"

# Activate environment
source "${ENV_PATH}/bin/activate"

# Prepare dataset if not already done
python "${REPO_DIR}/training/dataset.py" --root "${REPO_DIR}/datasets"

# Distributed training via torchrun
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

srun torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_DIR}/training/train.py" \
    --config "${REPO_DIR}/training/config.yaml"
