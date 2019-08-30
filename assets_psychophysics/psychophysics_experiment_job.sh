#!/bin/bash
#
#SBATCH --job-name=psychophysics
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=400
#SBATCH --time=0-00:10:00
#SBATCH --array=0-548
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_bernox2005_FixedFilter_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_oxenham2004_080to320Hz_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_AltPhase_v01_bestckpt.json' \
# -j ${job_idx}

python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
-r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_mooremoore2003_080to480Hz_bestckpt.json' \
-j ${job_idx}