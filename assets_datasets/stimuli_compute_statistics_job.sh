#!/bin/bash
#
#SBATCH --job-name=mean_spectrum
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-8:00:00
#SBATCH --time-min=0-8:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --partition=mcdermott

source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10_filter_signalHPv00/*.hdf5'

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab

python -u stimuli_compute_statistics.py \
-r "${source_fn_regex}" \
-k '/stimuli/signal'
