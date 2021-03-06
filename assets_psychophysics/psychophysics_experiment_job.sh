#!/bin/bash
#
#SBATCH --job-name=psychophysics
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36000
#SBATCH --time=0-6:00:00
#SBATCH --array=0-39
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --exclude=node[001-030]
#SBATCH --requeue
#SBATCH --dependency=afterok:20033349

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
echo $(hostname)

OUTDIR_REGEX='/om2/user/msaddler/tfauditoryutil/saved_models/PNDv08/*/arch_0???'
EFN_PREFIX='EVAL_SOFTMAX_'
PRIOR_RANGE='0.5'


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2/user/msaddler \
-B /om4 \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2/user/msaddler \
-B /om4 \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}transposedtones_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2/user/msaddler \
-B /om4 \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}altphase_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2/user/msaddler \
-B /om4 \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}freqshifted_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


singularity exec --nv \
-B /home \
-B /om \
-B /nobackup \
-B /om2/user/msaddler \
-B /om4 \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
-r "${OUTDIR_REGEX}/${EFN_PREFIX}mistunedharm_v01_bestckpt.json" \
-p ${PRIOR_RANGE} \
-j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v01_thresh40_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v01_noise08_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v01_noise10_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v01_dbspl85_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}transposedtones_v01_dbspl85_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_alt_phase.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}altphase_v01_dbspl85_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_freq_shifted.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}freqshifted_v01_dbspl85_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0experiment_mistuned_harmonics.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}mistunedharm_v01_dbspl85_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}testsnr_v01_bestckpt.json" \
# -k 'snr_per_component' \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}testspl_v01_bestckpt.json" \
# -k 'dbspl' \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}testsnr_v02_bestckpt.json" \
# -k 'snr_per_component' \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_generalized.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}testspl_v03_bestckpt.json" \
# -k 'dbspl' \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v02_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v03_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}lowharm_v04_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}

# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2/user/msaddler \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_transposed_tones.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}transposedtones_v02_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}

# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel10dB_harmlevel20dBSPL_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel40dB_harmlevel50dBSPL_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /nobackup \
# -B /om2 \
# -B /om4 \
# /om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
# python /om2/user/msaddler/pitchnet/assets_psychophysics/f0dl_bernox.py \
# -r "${OUTDIR_REGEX}/${EFN_PREFIX}bernox2006_TENlevel65dB_harmlevel75dBSPL_bestckpt.json" \
# -p ${PRIOR_RANGE} \
# -j ${job_idx}
