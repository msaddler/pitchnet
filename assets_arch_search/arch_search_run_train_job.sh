#!/bin/bash
#
#SBATCH --job-name=arch_search
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=18000
#SBATCH --gres=gpu:QUADRORTX6000:1
##SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --time=1-00:00:00
##SBATCH --time-min=0-24:00:00
##SBATCH --array=0-399
#SBATCH --array=37,41,50,67,78,80,82,84,87,150,152,153,154,156,157,160-166,168,170,172-177,179-182,184-187,189,190,193,197,318,325,329,331,333,335,337,338,354,357,358,361,365,374,388,398
##SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
SOURCE_CONFIG_FN='config_arch_search_v02.json'
OUTPUT_DIR_PATTERN="/saved_models/arch_search_v02/arch_{:04d}"
OUTPUT_LOG_FN=$(printf "$SCRATCH_PATH/pitchnet/saved_models/arch_search_v02/logs_train/arch_%04d.log" ${job_idx})

DATA_TRAIN='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords'
DATA_EVAL='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords'

echo $OUTPUT_LOG_FN
echo $(hostname)

singularity exec --nv \
-B /home \
-B /nobackup \
-B /om \
-B /om2 \
-B /om4 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B $SCRATCH_PATH/data_pitchnet:/data \
-B $SCRATCH_PATH/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_archsearch/arch_search_run_train.py \
-o "${OUTPUT_DIR_PATTERN}" \
-c "${SOURCE_CONFIG_FN}" \
-j ${job_idx} \
-dt $DATA_TRAIN \
-de $DATA_EVAL \
2>&1 | tee $OUTPUT_LOG_FN
