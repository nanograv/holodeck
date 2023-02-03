#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Run parallel Semi-Analytic Model (SAM) library generation script in parallel.
# Setup for running on the `minos` computer
# --------------------------------------------------------------------------------------------------

echo "================================================================================"
echo "==================    HOLODECK - run_gen_lib_sams.sh    ========================"
echo "================================================================================"
echo ""

OUTPUT="output/big-circ-01_2023-02-02_01_n2000_s40_r100_f40"
# OUTPUT=$OUTPUT"__TEST"
SCRIPT="scripts/gen_lib_sams.py"
LOG_NAME="job_log"
mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

cp $0 "$OUTPUT/"
LOG_OUT="$LOG_NAME.out"
LOG_ERR="$LOG_NAME.err"

mpirun -n 19  python $SCRIPT $OUTPUT -n 2000 -s 40 -r 100 -f 40  1> $LOG_OUT 2> $LOG_ERR &
# mpirun -n 2  python $SCRIPT $OUTPUT -n 4 -s 10 -r 10 -f 4  1> $LOG_OUT 2> $LOG_ERR &

