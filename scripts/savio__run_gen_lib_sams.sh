#!/bin/bash -l
#    run with `sbatch <SCRIPT>`

#SBATCH --account=fc_lzkhpc               # Allocation/Account 'fc_lzkhpc'
#SBATCH --job-name="astro-tight-02-gw"       # Name of job    `-J`
#SBATCH --mail-user=lzkelley@berkeley.edu   # Designate email address for job communications
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
###SBATCH --partition=savio_bigmem      # `savio_bigmem` 20 cores/node, allocation *by node*, 512 GB/node

#SBATCH --qos=savio_normal            # 24 nodes max per job, 72:00:00 wallclock limit
#SBATCH -t 48:00:00                   # Walltime/duration of the job  [HH:MM:SS]
#SBATCH --nodes=4
#SBATCH --ntasks=96                   # Number of MPI tasks
#SBATCH --ntasks-per-node=24          # expand memory
#SBATCH --mail-type=ALL               # {ALL, BEGIN, END, NONE, FAIL, REQUEUE}
# --------------------

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


# ====    setup parameters    ====

NAME="astro-tight-02-gw_2023-03-03"
PARS="n10000_s61-81-101_r100_f40"
SPACE="PS_Astro_Tight_02_GW"

#NAME="TEST_"$NAME

SCRIPT="./scripts/gen_lib_sams.py"
LOG_NAME=$NAME"_job-log"

OUTPUT="/global/scratch/users/lzkelley/holodeck_output/"$NAME"_"$PARS

mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

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

mpirun -np 96  python $SCRIPT $SPACE $OUTPUT -n 10000 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR
# mpirun -np 4  python $SCRIPT $SPACE $OUTPUT -n 8 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR


# ====    copy final products to share folder for uploading    ====

SHARE_OUTPUT=$OUTPUT"_SHARE"
mkdir -p $SHARE_OUTPUT
cp $OUTPUT/{*.py,holodeck*.log,*.hdf5} $SHARE_OUTPUT/
cp {$LOG_ERR,$LOG_OUT} $SHARE_OUTPUT/
