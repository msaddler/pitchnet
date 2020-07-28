#!/bin/bash
#
#SBATCH --job-name=pystraight
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --time-min=0-8:00:00
#SBATCH --exclude=node[001-030]
##SBATCH --partition=mcdermott
#SBATCH --array=0-99
#SBATCH --dependency=afterok:17395165

# source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08inst/noise_TLAS_snr_neg10pos10/*.hdf5'
# source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_mfcc/PNDv08matched12_TLASmatched12_snr_neg10pos10_phase0/*.hdf5'
source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_mfcc/PNDv08negated12_TLASmatched12_snr_neg10pos10_phase0/*.hdf5'

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab # Activate conda environment with "pystraight" installed
echo $(hostname)

python -u stimuli_analyze_pystraight.py \
-r "${source_fn_regex}" \
-d "PYSTRAIGHT_v01_foreground" \
-sks "stimuli/signal" \
-j $job_idx
