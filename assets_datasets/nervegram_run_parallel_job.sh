#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --time-min=0-36:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --array=0-19
##SBATCH --partition=mcdermott
#SBATCH --partition=use-everything
#SBATCH --requeue
##SBATCH --dependency=afterok:15515107

echo $(hostname)

## Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/PND_sr32000*.hdf5'
# dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order/bez2018meanrates.hdf5'
source_regex='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/*.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/sr20000_cf100_species002_spont070_BW10eN1_IHC6000Hz_IHC7order/bez2018meanrates.hdf5'

jobs_per_source_file=20
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab # Activate conda environment with "cython_bez2018" module installed

# python -u nervegram_run_parallel.py \
# -s "${source_regex}" \
# -d "${dest_filename}" \
# -j ${job_idx} \
# -jps ${jobs_per_source_file} \
# -bwsf '1.0' \
# -lpf '50.0' \
# -lpfo '7' \
# -sks 'stimuli/signal_in_noise' \
# -sksr 'sr' \
# -mrsr '20000.0' \
# -spont 'H'

python -u nervegram_run_parallel.py \
-s "${source_regex}" \
-d "${dest_filename}" \
-j ${job_idx} \
-jps ${jobs_per_source_file} \
-bwsf '1.0' \
-lpf '6000.0' \
-lpfo '7' \
-sks 'tone_in_noise' \
-sksr 'config_tone/fs' \
-mrsr '20000.0' \
-spont 'H'
