#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --error=./errors/gp-fitting-results-%J.err
#SBATCH --output=./outputs/gp-fitting-results-%J.out
#SBATCH --mail-user=<your-mail>
#SBATCH --mail-type=ALL
#SBATCH --job-name=gp-fitting

module purge
module load anaconda/anaconda3
source ~/.bashrc
conda activate holodeck

LIBRARY="./spec_libraries/big-circ-01_2023-02-09_01_n1000_s50_r100_f40/sam_lib.hdf5"
NFREQS=14
# Should use NWALKERS=CPUS*2
NWALKERS=36
NSAMPLES=1500
BURN_FRAC=0.25
TEST_FRAC=0.10
CENTER_MEASURE="median"
KERNEL="ExpSquaredKernel"

python train_gp.py  "${LIBRARY}" -f ${NFREQS} -w ${NWALKERS} -s ${NSAMPLES} -b ${BURN_FRAC} -t ${TEST_FRAC} -c "${CENTER_MEASURE}" -k "${KERNEL}"
