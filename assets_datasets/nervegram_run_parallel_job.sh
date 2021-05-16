#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --array=0-59
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue
##SBATCH --dependency=afterok:17389934

## Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/*.hdf5'
# jobs_per_source_file=3

source_regex='/om/user/msaddler/data_pitchnet/bernox2005/neurophysiology_v02_inharmonic_fixed_EqualAmpTEN_lharm01to15_phase0_f0min080_f0max640_seed???/stim.hdf5'
jobs_per_source_file=15

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab # Activate conda environment with "cython_bez2018" module installed
echo $(hostname)

dest_filename='sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order'
python -u nervegram_run_parallel.py \
-s "${source_regex}" \
-d "${dest_filename}" \
-j ${job_idx} \
-jps ${jobs_per_source_file} \
-bwsf '1.0' \
-lpf '3000.0' \
-lpfo '7' \
-sks 'auto' \
-sksr 'sr' \
-mrsr '20000.0' \
-spont '70.0' \
-ncf 100 \
-nst 1
