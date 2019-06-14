#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-0:30:00
#SBATCH --time-min=0-0:30:00
#SBATCH --array=0-0
#SBATCH --qos=mcdermott #use-everything

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex='/om/user/msaddler/data_pitchnet/SyntheticTonesLowpass/SyntheticTonesLowpass_v0_noiseJWSS*.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/SyntheticTonesLowpass/test.hdf5'
jobs_per_source_file=10000

python -u nervegram_run_parallel.py "${source_regex}" ${dest_filename} ${SLURM_ARRAY_TASK_ID} ${jobs_per_source_file}