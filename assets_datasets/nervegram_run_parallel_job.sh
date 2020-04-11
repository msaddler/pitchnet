#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --exclude=node[001-030,080]
#SBATCH --array=0-19
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue
##SBATCH --dependency=afterok:15515107

## Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)

# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10_filter_signalBPv02/PND_sr32000*.hdf5'
# dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10_filter_signalBPv02/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=3

# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_synthetic/noise_UMNm_snr_neg10pos10_phase01_filter_signalBPv00/PND_sr32000*.hdf5'
# dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_synthetic/noise_UMNm_snr_neg10pos10_phase01_filter_signalBPv00/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=1

# source_regex='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/sr20000_cf100_species002_spont070_BW30eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=20

source_regex='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01/*.hdf5'
dest_filename='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01/sr20000_cf100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order/bez2018meanrates.hdf5'
jobs_per_source_file=20

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab # Activate conda environment with "cython_bez2018" module installed
echo $(hostname)

# python -u nervegram_run_parallel.py \
# -s "${source_regex}" \
# -d "${dest_filename}" \
# -j ${job_idx} \
# -jps ${jobs_per_source_file} \
# -bwsf '1.0' \
# -lpf '3000.0' \
# -lpfo '7' \
# -sks 'stimuli/signal_in_noise' \
# -sksr 'sr' \
# -mrsr '20000.0' \
# -spont 'H'

# python -u nervegram_run_parallel.py \
# -s "${source_regex}" \
# -d "${dest_filename}" \
# -j ${job_idx} \
# -jps ${jobs_per_source_file} \
# -bwsf '3.0' \
# -lpf '3000.0' \
# -lpfo '7' \
# -sks 'tone_in_noise' \
# -sksr 'config_tone/fs' \
# -mrsr '20000.0' \
# -spont 'H'

python -u nervegram_run_parallel.py \
-s "${source_regex}" \
-d "${dest_filename}" \
-j ${job_idx} \
-jps ${jobs_per_source_file} \
-bwsf '1.0' \
-lpf '50.0' \
-lpfo '7' \
-sks 'stimuli/signal' \
-sksr 'config_tone/fs' \
-mrsr '20000.0' \
-spont 'H'
