#!/bin/bash
#SBATCH --job-name=HER2_MIL_Base
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

echo "========================================================="
echo "Inizio addestramento: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo assegnato: $SLURM_NODELIST"
echo "========================================================="

export TMPDIR=/scratch.hpc/leonardo.meloni/tmp

source /scratch.hpc/leonardo.meloni/miniconda3/bin/activate
conda activate her2_env

python script/classificatore_her2_from_wsi.py

echo "========================================================="
echo "Addestramento completato: $(date)"
echo "========================================================="