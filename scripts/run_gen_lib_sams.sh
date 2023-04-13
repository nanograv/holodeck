#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Run parallel Semi-Analytic Model (SAM) library generation script in parallel.
# Setup for running on the `minos` computer
# --------------------------------------------------------------------------------------------------

echo "================================================================================"
echo "==================    HOLODECK - run_gen_lib_sams.sh    ========================"
echo "==================         $(date +'%Y-%m-%d|%H:%M:%S')         ========================"
echo "================================================================================"
echo ""

# ====    setup simulation parameters    ====
NAME="ps-test-pd-01_2023-03-01"
PARS="n100_s61-81-101_r100_f40"

# NAME="TEST__"$NAME

SCRIPT="./scripts/gen_lib_sams.py"
LOG_NAME="job_log_"$NAME

OUTPUT="./output/"$NAME"_"$PARS

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
set -x

echo "Running python script $(date +'%Y-%m-%d|%H:%M:%S')"
echo ""

# mpirun -np 80  python $SCRIPT $OUTPUT -n 4000 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR
# mpirun -n 4  python $SCRIPT $OUTPUT -n 8 -s 10 -r 20 -f 40  1> $LOG_OUT 2> $LOG_ERR
python $SCRIPT $OUTPUT -n 8 -s 10 -r 20 -f 40  1> $LOG_OUT 2> $LOG_ERR

echo "Completed python script $(date +'%Y-%m-%d|%H:%M:%S')"
echo "Copying files"
echo ""

SHARE_OUTPUT=$OUTPUT"_SHARE"
mkdir -p $SHARE_OUTPUT
cp $OUTPUT/{*.py,holodeck*.log,*.hdf5} $SHARE_OUTPUT/
cp {$LOG_ERR,$LOG_OUT} $SHARE_OUTPUT/

echo ""
echo "==================         $(date +'%Y-%m-%d|%H:%M:%S')         ========================"
echo "================================================================================"
echo ""
