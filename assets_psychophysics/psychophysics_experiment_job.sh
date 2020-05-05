#!/bin/bash
#
#SBATCH --job-name=psychophysics
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36000
#SBATCH --time=0-6:00:00
#SBATCH --array=0-5
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --exclude=node[001-030]
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

OUTDIR_REGEX='/om2/user/msaddler/pitchnet/saved_models/models_sr20000/arch_0302/*'
# OUTDIR_REGEX='/om/scratch/*/msaddler/pitchnet/saved_models/arch_search_v01_spont*/arch*'
EFN_PREFIX='EVAL_SOFTMAX_TEST_*_ANMODEL_'
PRIOR_RANGE='0.5'


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2005_FixedFilter_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}oxenham2004_080to320Hz_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


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


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}mooremoore2003_080to480Hz_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}MistunedHarm_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}mcpherson2020_testSNR_v01_bestckpt.json" \
-k 'snr_per_component' \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}mcpherson2020_testSPL_v01_bestckpt.json" \
-k 'dbspl' \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel10dB_harmlevel20dBSPL_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel40dB_harmlevel50dBSPL_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2 \
-B /om4 \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel65dB_harmlevel75dBSPL_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}
