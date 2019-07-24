#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --nodes=1
#SBATCH --time=0-6:00:00
#SBATCH --time-min=0-3:00:00
#SBATCH --exclude=node[001-029]
#SBATCH --array=0-39
#SBATCH --qos=mcdermott
###SBATCH --qos=use-everything

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex='/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to500Hz/MooreMoore2003_frequencyShiftedComplexes_f0_080to500Hz.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to500Hz/cf100_species002_spont070/bez2018meanrates.hdf5'
jobs_per_source_file=40
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE

python -u nervegram_run_parallel.py "${source_regex}" "${dest_filename}" ${job_idx} ${jobs_per_source_file}
