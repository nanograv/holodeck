#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Run parallel Semi-Analytic Model (SAM) library generation script in parallel.
# Setup for running on the `minos` computer
# --------------------------------------------------------------------------------------------------

OUTPUT="output/simple01_2022-12-07_01"
SCRIPT="scripts/gen_lib_sams.py"
LOG_NAME="job_log"
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

cp $0 "$OUTPUT/"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"

mpirun -n 18  python $SCRIPT $OUTPUT -n 200 -s 50 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR &

cp $LOG_OUT "$OUTPUT/"
cp $LOG_ERR "$OUTPUT/"
