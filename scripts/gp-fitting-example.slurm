#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --error=./errors/gp-fitting-results-%J.err
#SBATCH --output=./outputs/gp-fitting-results-%J.out
#SBATCH --mail-user=<your-mail>
#SBATCH --mail-type=ALL
#SBATCH --job-name=gp-fitting

module purge
module load openmpi/<your-mpi-module>

module load anaconda/anaconda3
source ~/.bashrc
conda activate holodeck

CONFIG="./gp_config.ini"

mpiexec python gp_trainer.py "${CONFIG}"
