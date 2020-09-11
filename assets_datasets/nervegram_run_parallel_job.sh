#!/bin/bash
#
#SBATCH --job-name=bez2018model
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
##SBATCH --time-min=0-24:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --array=0-299
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue
##SBATCH --dependency=afterok:17389934

## Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)

# DIR_SOURCE="$SCRATCH_PATH"'/data_hearinglossnet/pitchrepnet_eval2afc_bernox2005_sr32000_dur150ms'
# DIR_DEST="$SCRATCH_PATH"'/data_hearinglossnet/pitchrepnet_eval2afc_bernox2005_sr32000_dur150ms'
# source_regex="$DIR_SOURCE"'/stim.hdf5'
# dest_filename="$DIR_DEST"'/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=50

# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_mfcc/PNDv08PYSnegated12_TLASmatched12_snr_neg10pos10_phase3/*.hdf5'
# dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_mfcc/PNDv08PYSnegated12_TLASmatched12_snr_neg10pos10_phase3/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=3

source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/PND_*.hdf5'
dest_filename="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
jobs_per_source_file=3

# source_regex='/om/user/msaddler/data_pitchnet/mcpherson2020/testSPL_v01_f0min080_f0max320_dBSPLmin000_dBSPLmax120/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/mcpherson2020/testSPL_v01_f0min080_f0max320_dBSPLmin000_dBSPLmax120/sr2000_cf1000_species002_spont070_BW10eN1_IHC0050Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=60

# source_regex='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01/sr2000_cf1000_species002_spont070_BW10eN1_IHC0050Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=60

# source_regex='/om/user/msaddler/data_pitchnet/neurophysiology/bernox2005_SlidingFixedFilter_lharm01to30_phase0_f0min080_f0max320/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/neurophysiology/bernox2005_SlidingFixedFilter_lharm01to30_phase0_f0min080_f0max320/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=15

# source_regex='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01_mUMN/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/moore1985/Moore1985_MistunedHarmonics_v01_mUMN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=40

# source_regex='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/*.hdf5'
# dest_filename='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates.hdf5'
# jobs_per_source_file=15


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
# -sks 'tone_in_noise' \
# -sksr 'config_tone/fs' \
# -mrsr '20000.0' \
# -spont '70' \
# -ncf 100 \
# -nst 1

python -u nervegram_run_parallel.py \
-s "${source_regex}" \
-d "${dest_filename}" \
-j ${job_idx} \
-jps ${jobs_per_source_file} \
-bwsf '1.0' \
-lpf '3000.0' \
-lpfo '7' \
-sks 'stimuli/signal_in_noise' \
-sksr 'sr' \
-mrsr '20000.0' \
-spont '0.1' \
-ncf 100 \
-nst 1

# python -u nervegram_run_parallel.py \
# -s "${source_regex}" \
# -d "${dest_filename}" \
# -j ${job_idx} \
# -jps ${jobs_per_source_file} \
# -bwsf '1.0' \
# -lpf '50.0' \
# -lpfo '7' \
# -sks 'stimuli/signal' \
# -sksr 'config_tone/fs' \
# -mrsr '2000.0' \
# -spont '70.0' \
# -ncf 100 \
# -nst 1
