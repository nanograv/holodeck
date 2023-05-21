#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --time=3:00:00
#SBATCH --array=0-14
#SBATCH --mem=16GB
#SBATCH --error=./errors/gp-fitting-results-%A_%a.err
#SBATCH --output=./outputs/gp-fitting-results-%A_%a.out
#SBATCH --mail-user=<your-mail>
#SBATCH --mail-type=ALL
#SBATCH --job-name=gp-fitting

module purge
module load openmpi/<your-mpi-module>

module load anaconda/anaconda3
source ~/.bashrc
conda activate holodeck

CONFIG="./gp_config.ini"

mpiexec -np 18 python gp_trainer.py "${CONFIG}" -f "${SLURM_ARRAY_TASK_ID}"
