#!/bin/bash
#
#SBATCH --job-name=arch_search
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=18000
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --time=0-48:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --array=0,23,31,37,41,7,51,53,54,55,56,62,63,64,65,68,70-76,101,103,106,109,112,113,117,119,123,124,127,130-133,136-139,142
#SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
SOURCE_CONFIG_FN='config_arch_search_v01.json'
OUTPUT_DIR_PATTERN="/saved_models/arch_search_v01/arch_{:04d}"
OUTPUT_LOG_FN=$(printf "/om/scratch/Fri/msaddler/pitchnet/saved_models/arch_search_v01/logs_train/arch_%04d.log" ${job_idx})

echo $OUTPUT_LOG_FN
echo $(hostname)

singularity exec --nv \
-B /home \
-B /nobackup \
-B /om \
-B /om2 \
-B /om4 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B /om/scratch/Fri/msaddler/data_pitchnet:/data \
-B /om/scratch/Fri/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/ibmHearingAid:/code_location \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python /om2/user/msaddler/pitchnet/assets_archsearch/arch_search_run_train.py \
-o "${OUTPUT_DIR_PATTERN}" \
-c "${SOURCE_CONFIG_FN}" \
-j ${job_idx} \
-dt /data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords \
-de /data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords \
2>&1 | tee $OUTPUT_LOG_FN
