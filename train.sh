#!/bin/bash

#SBATCH --job-name=NORT5
#SBATCH --account=project_465000157
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=7G
#SBATCH --cpus-per-task=7
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=pilot
#SBATCH --output=report/%j.out
#SBATCH --signal=B:TERM
#SBATCH --exclude=nid006851,nid006850,nid006949,nid005350,nid005136,nid007066,nid007069,nid006860,nid006988,nid006990,nid007096,nid005196,nid005197,nid007566,nid005138,nid005139,nid005552,nid005510,nid006767,nid006768,nid005515,nid005596,nid006105,nid007089,nid006857,nid007094,nid007339,nid005113,nid005122,nid005142,nid007118,nid007496,nid005146,nid005345,nid005554,nid007572,nid006772,nid006859,nid007064,nid007088,nid007120,nid005159,nid005346,nid005518,nid006774,nid006987,nid007067,nid007086,nid007338,nid007563,nid006512,nid007092,nid007337,nid005141,nid005599,nid007087,nid007113,nid005135,nid005195,nid005553,nid006108,nid007570,nid005511,nid005597,nid005653,nid006109,nid007114,nid007571,nid007573,nid005120,nid005351,nid006514,nid006863,nid007065,nid007097,nid007218,nid005424,nid006348,nid006357,nid006954,nid006959,nid005303,nid006106,nid006556,nid007503,nid005194,nid005198,nid007236,nid005604,nid005605,nid005952,nid005957,nid006377,nid006379,nid006760,nid006762,nid007196,nid007553,nid007556


set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# Load modules
module --quiet purge
module load LUMI/22.08
module load cray-python/3.9.12.1
module load rocm/5.0.2

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_VERBOSE=2

# export PYTHONUSERBASE='/projappl/project_465000157/.local'
# export PATH=$PYTHONUSERBASE/bin:$PATH
# export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH
source ../../pytorch_1.13.0/bin/activate

export WANDB_MODE=offline

#export TORCH_CPP_LOG_LEVEL=INFO
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export LOGLEVEL=INFO

trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

srun -W 0 python3 train.py "$@" &

PID="$!"
wait "${PID}"
