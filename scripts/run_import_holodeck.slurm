#!/bin/bash -l
#    run with `sbatch <SCRIPT>`

#SBATCH --account=fc_lzkhpc              # Allocation/Account 'fc_lzkhpc'
#SBATCH --job-name=ecg_test01            # Name of job    `-J`
#SBATCH --mail-user=emikogardiner@berkeley.edu           # Designate email address for job communications
#SBATCH --output=slurm-%x.%j.out            # Path for output must already exist   `-o`
#SBATCH --error=slurm-%x.%j.err             # Path for errors must already exist   `-e`

# ---- DEBUG ----
###SBATCH --partition=savio2_htc        # `savio2_htc` can use individual cores, instead of entire nodes (use for debugging)
###SBATCH --qos=savio_debug             # `savio_debug` :: 4 nodes max per job, 4 nodes in total, 00:30:00 wallclock limit
###SBATCH -t 00:30:00                   # Walltime/duration of the job  [HH:MM:SS]
###SBATCH --nodes=1
###SBATCH --ntasks=4                    # Number of MPI tasks
###SBATCH --mail-type=NONE              # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# ---------------

# ---- PRODUCTION ----
###SBATCH --partition=savio2            # `savio2` 24 cores/node, allocation *by node*, 64GB/node
#SBATCH --partition=savio2_bigmem     # `savio2_bigmem` 24 cores/node, allocation *by node*, 128 GB/node
#SBATCH --qos=savio_normal            # 24 nodes max per job, 72:00:00 wallclock limit
#SBATCH -t 48:00:00                   # Walltime/duration of the job  [HH:MM:SS]
#SBATCH --nodes=5
#SBATCH --ntasks=120                   # Number of MPI tasks
#SBATCH --ntasks-per-node=24          # expand memory
#SBATCH --mail-type=ALL               # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# --------------------

echo -e "\n====    "$0"    ====\n"

# ====    setup environment    ====

# setup linux modules
module purge
module load gcc openmpi python    # NOTE: replace with specific modules as needed
module list

# setup conda / python environment
source activate holo310           # NOTE: replace with appropriate anaconda environment
echo $PATH
conda info -e
which python
python --version


# ====    setup parameters    ====

SPACE="PS_Broad_Uniform_02B"           # NOTE: this must match exactly the class name in `holodeck/param_spaces.py`

# Specify the run name, these are used in creating the log-name and the output folder
NAME="broad-uniform-02b_2023-04-14"    # This is an arbitrary string specification for the job
PARS="n10000_s61-81-101_r100_f40"   # This is an additional arbitrary string specification for the job

# This is the script that will be run, make sure the path works
SCRIPT="./scripts/gen_lib_sams.py"
# This will be the name of the stdout and stderr logs
LOG_NAME=$NAME"_job-log"

# Output directory, make sure it has enough space
OUTPUT="/global/scratch/users/emikogardiner/holodeck_output/"$NAME"_"$PARS

# Create the output directory if it doesn't already exist
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

# copy this job script to the output directory
cp $0 "$OUTPUT/runtime_job-script.slurm"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"
echo "logs: ${LOG_OUT} ${LOG_ERR}"


# ====    run simulations    ====

echo "PWD:"
pwd
ls $SCRIPT
echo "Running `mpirun`"

set -x

# Real jobs
# mpirun -np 120  python $SCRIPT $SPACE $OUTPUT -n 10000 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR
# do something basic
mpirun -n 4 python -c “import holodeck”
# mpirun -n 4 python $SCRIPT -h

# Trivial test case to make sure the code runs
# mpirun -np 4  python $SCRIPT $SPACE $OUTPUT -n 8 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR


# ====    copy final products to share folder for uploading    ====

SHARE_OUTPUT=$OUTPUT"_SHARE"
mkdir -p $SHARE_OUTPUT
cp $OUTPUT/{*.py,holodeck*.log,*.hdf5} $SHARE_OUTPUT/
cp {$LOG_ERR,$LOG_OUT} $SHARE_OUTPUT/
