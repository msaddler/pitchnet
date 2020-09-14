#!/bin/bash
#SBATCH --job-name=pitchnet_train
#SBATCH --out="slurm-%A_%a.out"
##SBATCH --error="slurm-%A_%a.err"
##SBATCH --mail-user=msaddler@mit.edu
##SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

## Ensure `parallel` can be found
export PATH=$HOME/opt/bin:$PATH

parallel -j4 ./arch_search_run_train_satori.sh {1} {%} {#} ::: $(seq 145 149; seq 275 299)
