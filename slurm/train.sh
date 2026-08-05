#!/bin/bash
#SBATCH --job-name=HER2_MIL_Train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leonardo.meloni@unibo.it
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/leonardo.meloni/HER2
#SBATCH --output=slurm/log_output_%j.txt
#SBATCH --error=slurm/log_error_%j.txt

# Usage: sbatch slurm/train_job.sh path/to/config.yaml <features_run_id>
CONFIG=${1:-configs/default.yaml}
FEATURES_RUN=${2:?"Usage: sbatch train_job.sh <config.yaml> <features_run_id>"}

echo "========================================================="
echo "Training start: $(date)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Config: $CONFIG | Features run: $FEATURES_RUN"
echo "========================================================="


srun /scratch.hpc/leonardo.meloni/venv/bin/python3 /scratch.hpc/leonardo.meloni/HER2/script/train.py --config "$CONFIG" --features-run "$FEATURES_RUN"

echo "========================================================="
echo "Training done: $(date)"
echo "========================================================="
