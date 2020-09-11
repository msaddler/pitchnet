#!/bin/bash
#SBATCH --job-name=pitchnet_train
#SBATCH --out="slurm-%A_%a.out"
##SBATCH --error="slurm-%A_%a.err"
#SBATCH --mail-user=msaddler@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=18000
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --array=0-1

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
SOURCE_CONFIG_FN='config_arch_search_v02.json'
OUTPUT_DIR_PATTERN="/saved_models/arch_search_v02/arch_{:04d}"
OUTPUT_LOG_FN=$(printf "/nobackup/users/msaddler/pitchnet/saved_models/arch_search_v02/logs_train/arch_%04d.log" ${job_idx})

DATA_TRAIN='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-2]*.tfrecords'
DATA_EVAL='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[3-4]*.tfrecords'

echo $OUTPUT_LOG_FN
echo $(hostname)

singularity exec --nv \
-B /nobackup/users/msaddler/ \
-B /nobackup/users/msaddler/data_pitchnet:/data \
-B /nobackup/users/msaddler/pitchnet/saved_models:/saved_models \
-B /nobackup/users/msaddler/pitchnet/ibmHearingAid:/code_location \
docker://afrancl/ibm-hearing-aid-satori:tensorflow \
python -u /nobackup/users/msaddler/pitchnet/assets_archsearch/arch_search_run_train.py \
-o "${OUTPUT_DIR_PATTERN}" \
-c "${SOURCE_CONFIG_FN}" \
-j ${job_idx} \
-dt $DATA_TRAIN \
-de $DATA_EVAL \
2>&1 | tee $OUTPUT_LOG_FN
