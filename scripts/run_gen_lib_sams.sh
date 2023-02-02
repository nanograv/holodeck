#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Run parallel Semi-Analytic Model (SAM) library generation script in parallel.
# Setup for running on the `minos` computer
# --------------------------------------------------------------------------------------------------

OUTPUT="output/eccen-01_2023-01-31_01_n1000_s40_r100_f40"
#OUTPUT=$OUTPUT"__TEST"
SCRIPT="scripts/gen_lib_sams.py"
LOG_NAME="job_log"
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

cp $0 "$OUTPUT/"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"

mpirun -n 8  python $SCRIPT $OUTPUT  -n 100 -s 40 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR &
# mpirun -n 2  python $SCRIPT $OUTPUT -n 4 -s 40 -r 100 -f 10  1> $LOG_OUT 2> $LOG_ERR &

