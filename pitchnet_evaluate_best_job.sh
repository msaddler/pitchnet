#!/bin/bash
#
#SBATCH --job-name=pitchnet_eval
#SBATCH --out="/om2/user/msaddler/pitchnet/slurm_pitchnet_eval-%A_%a.out"
##SBATCH --gres=gpu:tesla-v100:1
#SBATCH --gres=gpu:QUADRORTX6000:1
##SBATCH --gres=gpu:GEFORCEGTX1080TI:1
##SBATCH --gres=gpu:GEFORCERTX2080TI:1
#SBATCH --mem=18000
#SBATCH --cpus-per-task=10
#SBATCH --time=0-02:00:00
##SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --array=0-9
##SBATCH --array=87,97,208,245,270,285,287,302,325,373

# ZERO_PADDED_JOBID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# OUTDIR='/saved_models/arch_search_v01_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_'$ZERO_PADDED_JOBID
# SAVED_MODELS_PATH="$SCRATCH_PATH/pitchnet/saved_models"

# ZERO_PADDED_JOBID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# OUTDIR='/saved_models/arch_search_v01/arch_'$ZERO_PADDED_JOBID
# SAVED_MODELS_PATH="$SCRATCH_PATH/pitchnet/saved_models"

OUTDIR='/saved_models/arch_0302_PND_v08_TLAS_snr_neg10pos10_AN_BW10eN1_IHC3000Hz_classification'$SLURM_ARRAY_TASK_ID
# OUTDIR='/saved_models/models_sr20000/arch_0302/PNDv08PYSnegated12_TLASmatched12_snr_neg10pos10_phase3_AN_BW10eN1_IHC3000Hz_classification'$SLURM_ARRAY_TASK_ID
SAVED_MODELS_PATH="/om2/user/msaddler/pitchnet/saved_models"

TFRECORDS_REGEX='sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/*.tfrecords'
EFN_PREFIX='EVAL_SOFTMAX_'
WRITE_PROBS_OUT=1

echo "evaluating model in output directory: $OUTDIR"
echo "evaluation data: $TFRECORDS_REGEX"

# <--- Start temporary script for de-bugging pitchrepnet_eval2afc_bernox2005 stimuli
singularity exec --nv \
-B /nobackup \
-B /home \
-B /om \
-B /om2/user/msaddler \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SAVED_MODELS_PATH:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/scratch/Sat/msaddler/data_hearinglossnet/pitchrepnet_eval2afc_bernox2005_sr32000_dur150ms/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_??????-??????.tfrecords" \
-efn "${EFN_PREFIX}pitchrepnet_eval2afc_bernox2005_sr32000_dur150ms_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT
# <--- END temporary script for de-bugging pitchrepnet_eval2afc_bernox2005 stimuli

# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}bernox2005_FixedFilter_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to480Hz/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mooremoore2003_080to480Hz_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}MistunedHarm_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}oxenham2004_080to320Hz_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/shackcarl1994/AltPhase_v01_f0min080_f0max320/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}AltPhase_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/mcpherson2020/testSNR_v01_f0min080_f0max320_dBSPLmin060_dBSPLmax060/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mcpherson2020_testSNR_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2/user/msaddler \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/mcpherson2020/testSPL_v01_f0min080_f0max320_dBSPLmin000_dBSPLmax120/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mcpherson2020_testSPL_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/bernox2006/nharm10_f0min100_f0max300_TENlevel10dB_harmlevel20dBSPL/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}bernox2006_TENlevel10dB_harmlevel20dBSPL_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/bernox2006/nharm10_f0min100_f0max300_TENlevel40dB_harmlevel50dBSPL/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}bernox2006_TENlevel40dB_harmlevel50dBSPL_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/bernox2006/nharm10_f0min100_f0max300_TENlevel65dB_harmlevel75dBSPL/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}bernox2006_TENlevel65dB_harmlevel75dBSPL_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# ############ RUN VALIDATION SET ############

# ZERO_PADDED_JOBID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# OUTDIR='/saved_models/IHC0050Hz_arch_search_v01_arch_0302_manipulations/arch_0302_'$ZERO_PADDED_JOBID
# SAVED_MODELS_PATH="$SCRATCH_PATH/pitchnet/saved_models"
# DATA_PATH="$SCRATCH_PATH/data_pitchnet"

# OUTDIR='/saved_models/models_sr20000/arch_0302/f0_label_024_PND_v08_TLAS_snr_neg10pos10_AN_BW10eN1_IHC3000Hz_classification'$SLURM_ARRAY_TASK_ID
# SAVED_MODELS_PATH="/om2/user/msaddler/pitchnet/saved_models"
# DATA_PATH="$SCRATCH_PATH/data_pitchnet"

# echo "evaluating model in output directory: $OUTDIR"
# echo "evaluation data: >>> validation set <<<"

# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B $DATA_PATH:/data \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords" \
# -efn "EVAL_validation_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 0 -vk 'f0_label_024:accuracy'
