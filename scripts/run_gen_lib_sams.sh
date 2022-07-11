#!/bin/bash
#    run with `sbatch run.s`

###SBATCH -A p30669               # Allocation
###SBATCH -p short                # Queue
#SBATCH -A b1094               # Allocation
#SBATCH -p ciera-std                # Queue
#SBATCH -t 48:00:00             # Walltime/duration of the job
###SBATCH -N 1                    # Number of Nodes
###SBATCH --ntasks-per-node=6     # Number of Cores (Processors) per Node
#SBATCH -n 32                    # Number of cores total
###SBATCH --mem=20G               # Total memory in GB needed for a job. Also see --mem-per-cpu
#SBATCH --mem-per-cpu=12GB

#SBATCH --mail-user=lzkelley@northwestern.edu   # Designate email address for job communications
#SBATCH --mail-type=NONE     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=test.out    # Path for output must already exist   `-o`
#SBATCH --error=test.err     # Path for errors must already exist   `-e`
#SBATCH --job-name="test_n32_m12gb"       # Name of job                          `-J`

# ========== Setup =============

# unload any modules that carried over from your command line session
module purge

# add a project directory to your PATH (if needed)
# export PATH=$PATH:/projects/p20XXX/tools/

# load modules you need to use
source ~/start.sh
module list

# ========== Run Code ==============
mpirun -n 32 python ./scripts/gen_lib_sams.py output/test_2022-06-27
