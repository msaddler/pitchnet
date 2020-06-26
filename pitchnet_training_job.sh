#!/bin/bash
#
#SBATCH --job-name=pitchnet_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=10
#SBATCH --mem=20000
#SBATCH --gres=gpu:tesla-v100:1
##SBATCH --gres=gpu:QUADRORTX6000:1
##SBATCH --gres=gpu:GEFORCEGTX1080TI:1
##SBATCH --gres=gpu:GEFORCERTX2080TI:1
#SBATCH --time=0-48:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --array=3-9
##SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

OUTDIR='/saved_models/models_sr20000/arch_0302/PND_v08spch_TLAS_snr_neg10pos10_AN_BW10eN1_IHC3000Hz_classification'$SLURM_ARRAY_TASK_ID
OUTPUT_LOG_FN='/om2/user/msaddler/pitchnet'$OUTDIR'/output_train.log'
DATA_TRAIN='/data/PND_v08spch/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords'
DATA_EVAL='/data/PND_v08spch/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords'

echo $OUTDIR
echo $OUTPUT_LOG_FN
echo $(hostname)

singularity exec --nv \
-B /home \
-B /om \
-B /om2/user/msaddler \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /code_location/multi_gpu/run_train_or_eval.py $OUTDIR \
-dt $DATA_TRAIN \
-de $DATA_EVAL \
-t -e -f \
2>&1 | tee $OUTPUT_LOG_FN