#!/bin/bash

#SBATCH --job-name=TOKENIZE
#SBATCH --output=no_tokenize.out
#SBATCH --account=project_465000157
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=small

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1

export PYTHONUSERBASE='/projappl/project_465000157/.local'
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist := "$SLURM_JOB_NODELIST
echo "Number of nodes := "$SLURM_JOB_NUM_NODES
echo "Ntasks per node := "$SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT := "$MASTER_PORT
echo "WORLD_SIZE := "$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR := "$MASTER_ADDR

srun -W 0 python3 no_tokenize.py
