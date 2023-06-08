#!/bin/bash -l
#    run with `sbatch <SCRIPT>`

#SBATCH --account=fc_lzkhpc               # Allocation/Account 'fc_lzkhpc'
#SBATCH --job-name="uni07-gw_new_s1000_r1000"       # Name of job    `-J`
#SBATCH --mail-user=lzkelley@berkeley.edu   # Designate email address for job communications
#SBATCH --output=slurm-%x.%j.out            # Path for output must already exist   `-o`
#SBATCH --error=slurm-%x.%j.err             # Path for errors must already exist   `-e`

# ---- DEBUG ----
###SBATCH --partition=savio2_htc        # `savio2_htc` can use individual cores, instead of entire nodes (use for debugging), cost=1.20
###SBATCH --qos=savio_debug             # `savio_debug` :: 4 nodes max per job, 4 nodes in total, 00:30:00 wallclock limit
###SBATCH -t 00:30:00                   # Walltime/duration of the job  [HH:MM:SS]
###SBATCH --nodes=1
###SBATCH --ntasks=4                    # Number of MPI tasks
###SBATCH --mail-type=NONE              # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# ---------------

# ---- PRODUCTION ----
#SBATCH --partition=savio2            # `savio2` 24 cores/node, allocation *by node*, 64GB/node, cost=0.75
#SBATCH --qos=savio_normal            # 24 nodes max per job, 72:00:00 wallclock limit
#SBATCH -t 48:00:00                   # Walltime/duration of the job  [HH:MM:SS]
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=24
#SBATCH --mail-type=ALL               # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# --------------------

NTASKS=120

echo -e "\n====    "$0"    ===="
echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"

# ====    setup parameters    ====

SPACE="PS_Uniform_07A"

NSAMPS=1000
NREALS=1000
NFREQS=40

# string specification for parameters
PARS="new_n${NSAMPS}_r${NREALS}_f${NFREQS}"
echo "pars: " $PARS

# Setup the name of the simulation and output directory
NAME=${SPACE,,}   # convert to lower-case
NAME=${NAME/ps_/}   # remove "PS_"
NAME=${NAME//_/-}  # replace all occurrences of '_' with '-'
DATE=$(date +'%Y-%m-%d')
NAME=$NAME"_"$PARS

NAME="TEST_"$NAME

echo "run name: " $NAME

# ====    setup environment    ====

module purge
module load gcc openmpi python
module list
source activate py310
echo $PATH
conda info -e
which python
python --version

SCRIPT="./scripts/gen_lib_sams.py"
LOG_NAME=$NAME"_job-log"

# Set desired path to output, directory will be created below
OUTPUT="/global/scratch/users/lzkelley/"$NAME

# Create output directory if needed
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

# Copy this script to the output directory
cp $0 "$OUTPUT/runtime_job-script.slurm"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"
echo "logs: ${LOG_OUT} ${LOG_ERR}"

# ====    run simulations    ====

echo "PWD:"
pwd
ls $SCRIPT
set -x

echo -e "Running mpiexec $(date +'%Y-%m-%d|%H:%M:%S')\n"
echo ""


# ---- Production Run ----
mpiexec -np $NTASKS  python $SCRIPT $SPACE $OUTPUT --gwb -n $NSAMPS -r $NREALS -f $NFREQS  1> $LOG_OUT 2> $LOG_ERR

# ---- DEBUG Run ----
# set the SAM shape very small, e.g. `-s 10` to run really fast
# mpiexec -np $NTASKS  python $SCRIPT $SPACE $OUTPUT -n $NSAMPS -r $NREALS -f $NFREQS -s 10  1> $LOG_OUT 2> $LOG_ERR

echo -e "Completed python script $(date +'%Y-%m-%d|%H:%M:%S')\n"


echo -e "Copying files\n"
# copy log files to the output directory
cp {$LOG_ERR,$LOG_OUT} $OUTPUT/

echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"
echo -e "============================\n"

