#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --nodes=1
#SBATCH --time=0-30:00:00
#SBATCH --time-min=0-24:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --array=0-49
#SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
##SBATCH --requeue
##SBATCH --dependency=afterok:14702742

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v04/noise_TLAS_snr_neg10pos03/augment_halflowpassnoise_v02/PND_sr32000_v04*.hdf5'
# dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_v04/noise_TLAS_snr_neg10pos03/augment_halflowpassnoise_v02/cf100_species002_spont070/bez2018meanrates.hdf5'
# source_regex='/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/oxenham2004/Oxenham2004_transposedTones_f0_080to320Hz/cf100_species002_spont070_lowpass1000Hz/bez2018meanrates.hdf5'
source_regex='/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to480Hz/*.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/mooremoore2003/MooreMoore2003_frequencyShiftedComplexes_f0_080to480Hz/cf100_species002_spont070_lowpass1000Hz/bez2018meanrates.hdf5'
jobs_per_source_file=50
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab # Activate conda environment with "cython_bez2018" module installed

python -u nervegram_run_parallel.py "${source_regex}" "${dest_filename}" ${job_idx} ${jobs_per_source_file}
