#!/bin/bash
#SBATCH --job-name=olmd-wiring-fix
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME            # your CINECA project account (saldo -b)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=olmd-wiring-fix-%j.out

# CINECA Leonardo booster defaults (A100 64GB). Adjust partition/account for
# other CINECA machines. The trainer checkpoints every epoch and resumes from
# latest.pt, so if the 24 h walltime ends first, simply resubmit this script.
#
# One-time staging (login node, has internet -- see README "CINECA retrain"):
#   $WORK/olmd/data      annotations under rwth-phoenix-2014/raw/...
#   $WORK/olmd/cache     phoenix14/*.npz (6841 files, 9.6 GiB)
#   $WORK/olmd/torch     hub/checkpoints/resnet34-b627a593.pth
#   $WORK/olmd/venv      python -m venv + pip install torch torchvision numpy pillow

set -euo pipefail

BASE="${WORK:?}/olmd"
export TORCH_HOME="$BASE/torch"
source "$BASE/venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"
srun python train_cineca.py \
    --data-root "$BASE/data" \
    --cache-root "$BASE/cache" \
    --output "$BASE/output/phoenix14-wiring-fix" \
    --epochs 70 --workers 8 --seed 0
