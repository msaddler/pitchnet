#!/bin/bash
#
#SBATCH --job-name=psychophysics
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=400
#SBATCH --time=0-01:10:00
#SBATCH --array=0-10
#SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

OUTDIR_REGEX='/om2/user/msaddler/pitchnet/saved_models/models_RSB/arch_0628/PND_v04_TLAS*'

# source activate mdlab

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_bernox2005_FixedFilter_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_oxenham2004_080to320Hz_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_AltPhase_v01_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_mooremoore2003_080to480Hz_bestckpt.json' \
# -j ${job_idx}

# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
# -r '/om/scratch/Mon/msaddler/pitchnet/saved_models/arch_search_v00/*/EVAL_MistunedHarm_v00_bestckpt.json' \
# -j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "$OUTDIR_REGEX/EVAL_bernox2005_FixedFilter_bestckpt.json" \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
-r "$OUTDIR_REGEX/EVAL_oxenham2004_080to320Hz_bestckpt.json" \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
-r "$OUTDIR_REGEX/EVAL_AltPhase_v01_bestckpt.json" \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
-r "$OUTDIR_REGEX/EVAL_mooremoore2003_080to480Hz_bestckpt.json" \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
-r "$OUTDIR_REGEX/EVAL_MistunedHarm_v00_bestckpt.json" \
-j ${job_idx}
