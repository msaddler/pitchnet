#!/bin/bash
#
#SBATCH --job-name=pitchnet_eval
#SBATCH --out="/om2/user/msaddler/pitchnet/slurm-%A_%a.out"
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=4
#SBATCH --time=0-2:00:00
##SBATCH --exclude=node063
#SBATCH --partition=mcdermott
#SBATCH --array=0-2

OUTDIR='/saved_models/arch_0628/PND_v04_JWSS_classification'$SLURM_ARRAY_TASK_ID
echo "evaluating model in output directory: $OUTDIR"

singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de '/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_bernox2005_FixedFilter_bestckpt.json' \
-o "$OUTDIR"


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de '/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to480Hz/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_mooremoore2003_080to480Hz_bestckpt.json' \
-o "$OUTDIR"


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de '/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v00/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_MistunedHarm_v00_bestckpt.json' \
-o "$OUTDIR"


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de '/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_oxenham2004_080to320Hz_bestckpt.json' \
-o "$OUTDIR"


singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python pitchnet_evaluate_best.py \
-de '/om/user/msaddler/data_pitchnet/shackcarl1994/AltPhase_v01_f0min080_f0max320/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_AltPhase_v01_bestckpt.json' \
-o "$OUTDIR"
