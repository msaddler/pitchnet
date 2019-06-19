#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --time-min=0-10:30:00
#SBATCH --exclude=node[001-029]
#SBATCH --array=0-99
###SBATCH --qos=mcdermott #use-everything

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex='/om/user/msaddler/data_pitchnet/bernox2005/SyntheticTonesBandpass/SyntheticTonesBandpass_v0_noiseJWSS.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/bernox2005/SyntheticTonesBandpass/cf050_species002_spont070/bez2018meanrates.hdf5'
jobs_per_source_file=100
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

python -u nervegram_run_parallel.py "${source_regex}" ${dest_filename} ${job_idx} ${jobs_per_source_file}
