#!/bin/bash

#
# nohup mpirun -np 16 holodeck_lib_gen -n 64 -r 100 --gwb --ss --params PS_Astro_Strong_Hard_Only ./output/astro-strong-hard-only &
#
# nohup bash run_holodeck_lib_gen__minos.sh &
#

# ====    setup parameters    ====

SPACE="PS_Astro_Strong_All"

NTASKS=4

NSAMPS=20
NREALS=10
NFREQS=40
NLOUDEST=3

PARS="n${NSAMPS}_r${NREALS}_f${NFREQS}"

echo "pars: " $PARS

echo -e "\n====    "$0"    ===="
echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"

NAME=${SPACE,,}   # convert to lower-case
NAME=${NAME/ps_/}   # remove "PS_"
NAME=${NAME//_/-}  # replace all occurrences of '_' with '-'
DATE=$(date +'%Y-%m-%d')
# NAME=$NAME"_"$PARS"_domain"
NAME=$NAME"_"$PARS"_doppler-test"

echo "run name: " $NAME

# ====    setup environment    ====

COMMAND="holodeck_lib_gen"
LOG_NAME=$NAME"_job-log"

OUTPUT="./output/"$NAME

mkdir -p $OUTPUT
echo "Output directory: ${OUTPUT}"

cp $0 "$OUTPUT/runtime_job-script"
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

# START NEW
# mpirun -np $NTASKS  $COMMAND $SPACE $OUTPUT -n $NSAMPS -r $NREALS -f $NFREQS  1> $LOG_OUT 2> $LOG_ERR &
mpiexec -np $NTASKS  $COMMAND $SPACE $OUTPUT -n $NSAMPS -r $NREALS -f $NFREQS  1> $LOG_OUT 2> $LOG_ERR &

# RESUME
# mpirun -np $NTASKS  $COMMAND $SPACE $OUTPUT --resume -n $NSAMPS -r $NREALS -f $NFREQS  1> $LOG_OUT 2> $LOG_ERR &

# DOMAIN
# mpirun -np $NTASKS  $COMMAND $SPACE $OUTPUT --domain -n $NSAMPS -r $NREALS -f $NFREQS -l $NLOUDEST  1> $LOG_OUT 2> $LOG_ERR &


echo -e "Completed python script $(date +'%Y-%m-%d|%H:%M:%S')\n"

# ====    copy final products to share folder for uploading    ====

echo -e "Copying files\n"

cp {$LOG_ERR,$LOG_OUT} $OUTPUT/

echo -e "====    $(date +'%Y-%m-%d|%H:%M:%S')    ====\n"
echo -e "============================\n"

