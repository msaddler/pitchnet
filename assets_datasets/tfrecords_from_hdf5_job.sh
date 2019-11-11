#!/bin/bash
#
#SBATCH --job-name=tfrecords_from_hdf5
#SBATCH --out="trash/tfrecords_slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2500
#SBATCH --nodes=1
#SBATCH --time=0-1:40:00
#SBATCH --array=0-299%50
##SBATCH --partition=mcdermott
#SBATCH --partition=use-everything
##SBATCH --dependency=afterok:14990544
##SBATCH --exclude=node[031-077]

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v04/noise_TLAS_snr_neg10pos03/sr16000_cf100_species002_spont070_BW10eN1_IHC7000Hz_IHC7order/bez2018meanrates_*.hdf5'
# source_regex='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/cf100_species002_spont070_BW*/bez2018meanrates_*.hdf5'
jobs_per_source_file=1
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

module add openmind/singularity

singularity exec --nv \
-B /om \
-B /om2 \
-B /nobackup \
/om2/user/msaddler/singularity-images/tensorflow-1.13.1-gpu-py3.img \
python -u tfrecords_from_hdf5.py "${source_regex}" ${job_idx} ${jobs_per_source_file}
