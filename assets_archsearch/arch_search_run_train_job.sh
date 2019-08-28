#!/bin/bash
#
#SBATCH --job-name=arch_search
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
####SBATCH --gres=gpu:GEFORCERTX2080:1
####SBATCH --gres=gpu:1 --constraint=high-capacity
#SBATCH --time=0-48:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --array=703,708,724,730
#SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
SOURCE_CONFIG_FN='example_config.json'
OUTPUT_DIR_PATTERN="/saved_models/arch_search_v00/arch_{:04d}"
OUTPUT_LOG_FN=$(printf "/om/scratch/Fri/msaddler/pitchnet/saved_models/arch_search_v00/logs_train/arch_%04d.log" ${job_idx})

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
/om/user/francl/ibm_hearing_aid/tfv1.13_openmind.simg \
python /om2/user/msaddler/pitchnet/assets_archsearch/arch_search_run_train.py \
-o "${OUTPUT_DIR_PATTERN}" \
-c "${SOURCE_CONFIG_FN}" \
-j ${job_idx} \
-dt /data/PND_v04/noise_JWSS_snr_neg10pos03/cf100_species002_spont070/bez2018meanrates_0[0-7]*.tfrecords \
-de /data/PND_v04/noise_JWSS_snr_neg10pos03/cf100_species002_spont070/bez2018meanrates_0[8-9]*.tfrecords \
2>&1 | tee $OUTPUT_LOG_FN
