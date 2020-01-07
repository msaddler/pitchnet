#!/bin/bash
#
#SBATCH --job-name=synthetic_tones
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-2:00:00
#SBATCH --time-min=0-1:00:00
##SBATCH --exclude=node[001-030]
#SBATCH --array=0-99
##SBATCH --partition=mcdermott

dest_filename="$SCRATCH_PATH/data_pitchnet/PND_synthetic/noise_UMNm_snr_neg10pos10_phase03_filter_signalLPv02/PND_sr32000_LPv02.hdf5"
num_parallel_jobs=100
num_total_stimuli=700000
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
source activate mdlab

python -u stimuli_generate_synthetic_tones.py \
-d "${dest_filename}" \
-j ${job_idx} \
-npj ${num_parallel_jobs} \
-nts ${num_total_stimuli}
