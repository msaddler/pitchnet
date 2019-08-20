#!/bin/bash
#
#SBATCH --job-name=tfrecords_from_hdf5
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2500
#SBATCH --nodes=1
#SBATCH --time=0-0:40:00
#SBATCH --array=0-35
#SBATCH --partition=sched_om_MCDERMOTT
##SBATCH --partition=use-everything
###SBATCH --dependency=afterok:12702768

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex='/om/user/msaddler/data_pitchnet/shackcarl1994/AltPhase_v01_f0min080_f0max320/cf100_species002_spont070/bez2018meanrates_*.hdf5'
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
