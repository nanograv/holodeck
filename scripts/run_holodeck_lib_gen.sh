#!/bin/bash -l
# ==================================================================================================
# Example job script to run holodeck library generation
# -----------------------------------------------------
#
# Excute this script on a personal computer using:   ``bash run_holodeck_lib_gen.sh``,
# or on a server with slurm scheduler using:       ``sbatch run_holodeck_lib_gen.sh``.
#
# The library generation is performed using the `holodeck.librarian.lib_gen` script.
# When holodeck is installed, this is aliased to the command-line function `holodeck_lib_gen`.
# The following two commands should be identical:
#
#     python -m holodeck.librarian.lib_gen  <ARGS>
#     holodeck_lib_gen  <ARGS>
#
# The library generation script is parallelized using MPI (specifically mpi4py in python).
# The actual run command can be very simple, for example:
#
#     mpirun -np 16 holodeck_lib_gen -n 64 -r 100 --gwb --ss --params PS_Classic_Phenom ./output/
#
# When running on a remote server with a job scheduler (e.g. SLURM), things can be a bit more
# complicated.  Example SLURM configuration is included below, but if you're running on a personal
# computer you can ignore these elements.
#
# Remember to use `nohup` and `&` to run in the background on remote server, so that it's not killed
# when you logout, e.g.
#
#     nohup  mpirun -np 16 holodeck_lib_gen -n 64 -r 100 --gwb --ss --params PS_Classic_Phenom ./output/  &
#
# ------------------
# Luke Zoltan Kelley
# LZKelley@berkeley.edu
# ==================================================================================================


# ====    SLURM job scheduler configuration    ====
# NOTE: SLURM uses a single '#' to denote its configuration.  Multiple '#' marks are ignored.

#SBATCH --account=fc_lzkhpc                        # Allocation/Account 'fc_lzkhpc'
#SBATCH --job-name="phenom_uniform_s1000_r1000"    # Name of job    `-J`
#SBATCH --mail-user=lzkelley@berkeley.edu          # Designate email address for job communications
#SBATCH --output=slurm-%x.%j.out                   # Path for stdout (dir must already exist!)
#SBATCH --error=slurm-%x.%j.err                    # Path for stderr (dir must already exist!)

# ---- DEBUG configuration
###SBATCH --partition=savio2_htc        # `savio2_htc` allocations by individual core, cost=1.20
###SBATCH --qos=savio_debug             # `savio_debug` 4 nodes max per job, 00:30:00 time limit
###SBATCH -t 00:30:00                   # Walltime/duration of the job  [HH:MM:SS]
###SBATCH --nodes=1                     # Number of nodes requested
###SBATCH --ntasks=4                    # Number of MPI tasks
###SBATCH --mail-type=NONE              # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# ---------------

# ---- PRODUCTION configuration
#SBATCH --partition=savio2            # `savio2` 24 cores/node, 64GB/node, cost=0.75
#SBATCH --qos=savio_normal            # 24 nodes max per job, 72:00:00 wallclock limit
#SBATCH -t 48:00:00                   # Walltime/duration of the job  [HH:MM:SS]
#SBATCH --nodes=5                     # Number of nodes requested
#SBATCH --ntasks-per-node=24
#SBATCH --mail-type=ALL               # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# --------------------

# ====    setup parameters    ====

# Name of the parameter space to be run.
# (this must match (including case) one of the holodeck `_Param_Space` subclasses)
SPACE="PS_Classic_Phenom_Uniform"
# Number of tasks/cores to run on
NTASKS=16
# Number of sample points in the latin hypercube sampling of parameter space
NSAMPS=512
# Number of realizations at each parameter sample ppint
NREALS=100
# Number of frequencies at which to calculate GW signals
NFREQS=40
# Set additional arguments
ARGS="--gwb --ss --params"

# Construct a string that encodes basic run-time parameters
PARS="n${NSAMPS}_r${NREALS}_f${NFREQS}"
echo "pars: " $PARS

echo -e "\n====    "$0"    ===="
echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"

# Construct a name of this run, combing the parameter-space name with the basic parameters above
NAME=${SPACE,,}     # convert to lower-case
NAME=${NAME/ps_/}   # remove "PS_" from the class name
NAME=${NAME//_/-}   # replace all occurrences of '_' with '-'
NAME=$NAME"_"$PARS

# Also append the current date to the run name
# DATE=$(date +'%Y-%m-%d')
# NAME=$NAME"_"$DATE

echo "run name: " $NAME


# ====    setup environment    ====

# ---- Load required modules when working on linux servers
# module purge
# module load gcc openmpi python
# module list

# ---- Load conda environment
# conda activate py311
# source activate py311
# echo $PATH
# conda info -e
# which python
# python --version

#COMMAND="python -m holodeck.librarian.gen_lib"
COMMAND="holodeck_lib_gen"
LOG_NAME=$NAME"_job-log"

# Choose the output path where files will be saved (this directory is created if it doesn't exist)
OUTPUT="./output/"$NAME
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

# Copy this script to the output directory
cp $0 "$OUTPUT/runtime_job-script"

# Construct names for the log files (stdout and stderr)
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"
echo "logs: ${LOG_OUT} ${LOG_ERR}"


# ====    run simulations    ====

echo "PWD:"
pwd
ls $SCRIPT
set -x

echo -e "Running mpirun $(date +'%Y-%m-%d|%H:%M:%S')\n"
echo ""

# this is the actual call to run holodeck
mpirun -np $NTASKS  $COMMAND $SPACE $OUTPUT  -n $NSAMPS -r $NREALS -f $NFREQS $ARGS  1> $LOG_OUT 2> $LOG_ERR

echo ""
echo -e "Completed python script $(date +'%Y-%m-%d|%H:%M:%S')\n"


# ====    copy log files to output directory    ====

echo -e "Copying files\n"

cp {$LOG_ERR,$LOG_OUT} $OUTPUT/

echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"
echo -e "============================\n"

