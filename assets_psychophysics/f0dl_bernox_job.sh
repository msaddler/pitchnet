#!/bin/bash
#
#SBATCH --job-name=f0dl_bernox
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=0-00:20:00
#SBATCH --array=40-548
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_bernox2005_FixedFilter_bestckpt.json' \
-j ${job_idx}
