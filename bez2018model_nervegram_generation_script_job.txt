#!/bin/bash
#
#SBATCH --job-name=nervegram_generation
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-3:00:00
#SBATCH --time-min=1:30:00
#SBATCH --array=0-96
#SBATCH --qos=mcdermott

python -u bez2018model_nervegram_generation_script.py ${SLURM_ARRAY_TASK_ID} 500 0