#!/bin/bash -l
#    run with `sbatch <SCRIPT>`

#SBATCH --account=fc_lzkhpc               # Allocation/Account 'fc_lzkhpc'
#SBATCH --job-name="big-circ-01"       # Name of job    `-J`

###SBATCH --partition=savio2_htc  # `savio2_htc` can use individual cores, instead of entire nodes (use for debugging)
###SBATCH --qos=savio_debug       # `savio_debug` :: 4 nodes max per job, 4 nodes in total, 00:30:00 wallclock limit
###SBATCH -t 00:20:00             # Walltime/duration of the job  [HH:MM:SS]
###SBATCH --nodes=1
###SBATCH --ntasks=4              # Number of MPI tasks

###SBATCH --partition=savio2            # `savio2` 24 cores/node, allocation *by node*, 64GB/node
###SBATCH --partition=savio2_bigmem       # `savio2_bigmem` 24 cores/node, allocation *by node*, 128 GB/node
#SBATCH --partition=savio_bigmem       # `savio_bigmem` 20 cores/node, allocation *by node*, 512 GB/node
#SBATCH --qos=savio_normal      # 24 nodes max per job, 72:00:00 wallclock limit
#SBATCH -t 20:00:00             # Walltime/duration of the job  [HH:MM:SS]
#SBATCH --nodes=4
#SBATCH --ntasks=80            # Number of MPI tasks

###SBATCH --nodes=1
###SBATCH --ntasks-per-node=4
###SBATCH --cpus-per-task=1

#SBATCH --mail-user=lzkelley@berkeley.edu   # Designate email address for job communications
###SBATCH --mail-type=ALL                     # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
#SBATCH --output=slurm-%x.%j.out            # Path for output must already exist   `-o`
#SBATCH --error=slurm-%x.%j.err             # Path for errors must already exist   `-e`

echo -e "\n====    savio__run_gen_lib_sams.sh    ===="
echo $0


# ====    setup environment    ====

module purge
module load gcc openmpi python
module list
source activate py310
echo $PATH
conda info -e
which python
python --version


# ====    setup simulation parameters    ====

OUTPUT="./output/big-circ-01_2023-02-09_01_n1000_s50_r100_f40"
# OUTPUT=$OUTPUT"__TEST"

SCRIPT="./scripts/gen_lib_sams.py"
LOG_NAME="job_log"
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

cp $0 "$OUTPUT/"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"
echo "logs: ${LOG_OUT} ${LOG_ERR}"

# ====    run simulations    ====

echo "PWD:"
pwd
ls $SCRIPT
echo "Running `mpirun`"

set -x

mpirun -np 20  python $SCRIPT $OUTPUT -n 1000 -s 50 -r 200 -f 40  1> $LOG_OUT 2> $LOG_ERR
# mpirun -np 4  python $SCRIPT $OUTPUT -n 8 -s 10 -r 10 -f 10  1> $LOG_OUT 2> $LOG_ERR
