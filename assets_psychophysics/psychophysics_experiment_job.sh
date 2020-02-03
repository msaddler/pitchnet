#!/bin/bash
#
#SBATCH --job-name=psychophysics
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000
#SBATCH --time=0-02:00:00
#SBATCH --array=0-391
#SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --exclude=node[001-030]
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

# OUTDIR_REGEX='/om2/user/msaddler/pitchnet/saved_models/models_sr20000/arch_0302/*'
OUTDIR_REGEX='/om/scratch/Wed/msaddler/pitchnet/saved_models/arch_search_v01/arch*'
EFN_PREFIX='EVAL_SOFTMAX_*'
PRIOR_RANGE='0.5'


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2005_FixedFilter_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}oxenham2004_080to320Hz_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}AltPhase_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}mooremoore2003_080to480Hz_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}MistunedHarm_v00_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}
