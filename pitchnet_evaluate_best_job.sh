#!/bin/bash
#
#SBATCH --job-name=pitchnet_eval
#SBATCH --out="/om2/user/msaddler/pitchnet/slurm-%A_%a.out"
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=0-3:00:00
##SBATCH --exclude=node063,node066
##SBATCH --partition=mcdermott
#SBATCH --array=500-750

# OUTDIR='/saved_models/PND_v04_TLAS_classification'$SLURM_ARRAY_TASK_ID
ZERO_PADDED_JOBID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
OUTDIR='/saved_models/arch_search_v00/arch_'$ZERO_PADDED_JOBID

TFRECORDS_REGEX='cf100_species002_spont070/*.tfrecords'

EFN_PREFIX='EVAL_SOFTMAX_'
WRITE_PROBS_OUT=1

echo "evaluating model in output directory: $OUTDIR"
echo "evaluation data: $TFRECORDS_REGEX"


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/$TFRECORDS_REGEX" \
-efn "${EFN_PREFIX}bernox2005_FixedFilter_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to480Hz/$TFRECORDS_REGEX" \
-efn "${EFN_PREFIX}mooremoore2003_080to480Hz_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v00/$TFRECORDS_REGEX" \
-efn "${EFN_PREFIX}MistunedHarm_v00_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz/$TFRECORDS_REGEX" \
-efn "${EFN_PREFIX}oxenham2004_080to320Hz_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de "/om/user/msaddler/data_pitchnet/shackcarl1994/AltPhase_v01_f0min080_f0max320/$TFRECORDS_REGEX" \
-efn "${EFN_PREFIX}AltPhase_v01_bestckpt.json" \
-o "$OUTDIR" \
-wpo $WRITE_PROBS_OUT



# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SCRATCH_PATH/data_pitchnet:/data \
# -B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
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
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SCRATCH_PATH/data_pitchnet:/data \
# -B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
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
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SCRATCH_PATH/data_pitchnet:/data \
# -B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v00/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}MistunedHarm_v00_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SCRATCH_PATH/data_pitchnet:/data \
# -B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}oxenham2004_080to320Hz_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT


# singularity exec --nv \
# -B /home \
# -B /om \
# -B /om2 \
# -B /om2/user/msaddler/python-packages:/python-packages \
# -B $SCRATCH_PATH/data_pitchnet:/data \
# -B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
# -B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
# /om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
# python pitchnet_evaluate_best.py \
# -de "/om/user/msaddler/data_pitchnet/shackcarl1994/AltPhase_v01_f0min080_f0max320/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}AltPhase_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT
