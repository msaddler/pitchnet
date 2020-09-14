#!/bin/bash
#SBATCH --job-name=pitchnet_train
#SBATCH --out="slurm-%j.out"
##SBATCH --error="slurm-%j.err"
##SBATCH --mail-user=msaddler@mit.edu
##SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --exclusive

## Create file containing SLURM node list
SLURM_TASK_LOG_FILENAME="slurm-tasklist-${SLURM_JOB_ID}.out"
SLURM_JOB_NODELIST_FILENAME="slurm-nodelist-${SLURM_JOB_ID}.out" 
# echo $SLURM_JOB_NODELIST | sed s/\,/\\n/g > $SLURM_JOB_NODELIST_FILENAME
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2}' | sed 's/,$//' > $SLURM_JOB_NODELIST_FILENAME

echo "==========================================================="
echo "wd=$(pwd)"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST_FILENAME=$SLURM_JOB_NODELIST_FILENAME"
echo "SLURM_TASK_LOG_FILENAME=$SLURM_TASK_LOG_FILENAME"
echo "==========================================================="

## Ensure `parallel` can be found
export PATH=$HOME/opt/bin:$PATH

## Define argument string for running `pitchnet_run_satori.sh` via parallel
PARALLEL_ARGUMENT_STRING='{1} $((({%}-1) / '$SLURM_JOB_NUM_NODES')) {#} {%}'
echo "PARALLEL_ARGUMENT_STRING=$PARALLEL_ARGUMENT_STRING"

## Execute `pitchnet_run_satori.sh` script with parallel arguments
parallel \
-j $SLURM_NTASKS_PER_NODE \
-k \
--slf $SLURM_JOB_NODELIST_FILENAME \
--joblog $SLURM_TASK_LOG_FILENAME \
--wd $(pwd) \
./arch_search_run_train_satori.sh $PARALLEL_ARGUMENT_STRING ::: 111 128 $(seq 135 144; seq 235 249; seq 275 299)
