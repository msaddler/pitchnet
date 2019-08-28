#!/bin/bash
#
#SBATCH --job-name=eval_arch_search
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
####SBATCH --gres=gpu:GEFORCERTX2080:1
####SBATCH --gres=gpu:1 --constraint=high-capacity
#SBATCH --time=0-2:00:00
#SBATCH --array=10-249
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
OUTPUT_DIR=$(printf "/saved_models/arch_search_v00/arch_%04d" ${job_idx})
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
python /om2/user/msaddler/pitchnet/assets_archsearch/arch_search_run_eval.py \
-de '/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/cf100_species002_spont070/*.tfrecords' \
-efn 'EVAL_bernox2005_FixedFilter_bestckpt.json' \
-o "${OUTPUT_DIR}"
