#!/bin/bash
#SBATCH --job-name=olmd-wiring-fix
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D33_168
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=olmd-wiring-fix-%j.out

# CINECA Leonardo booster (A100 64GB). The trainer checkpoints every epoch and
# resumes from latest.pt, so if the 24 h walltime ends first, simply resubmit.
#
# Environment: cineca-ai/4.3.0 supplies python 3.11 + torch 2.2 (CUDA 12.1) +
# torchvision 0.17 + numpy + pillow. No venv or pip needed.
#
# One-time staging under $BASE (login node, has internet -- see README):
#   $BASE/data    annotations under rwth-phoenix-2014/raw/...
#   $BASE/cache   phoenix14/*.npz (6841 files, 9.7 GiB)
#   $BASE/torch   hub/checkpoints/resnet34-b627a593.pth
#   $BASE/code    olmd.py data.py evaluate.py train_cineca.py (pinned commit
#                 in PROVENANCE.txt)

set -euo pipefail

BASE="/leonardo_work/EUHPC_D33_168/olmd"
export TORCH_HOME="$BASE/torch"

module purge
module load profile/deeplrn cineca-ai/4.3.0

cd "$BASE/code"
srun python3 train_cineca.py \
    --data-root "$BASE/data" \
    --cache-root "$BASE/cache" \
    --output "$BASE/output/phoenix14-wiring-fix" \
    --epochs 70 --workers 8 --seed 0
