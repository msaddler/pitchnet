#!/bin/bash
#
#SBATCH --job-name=mean_spectrum
##SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-8:00:00
#SBATCH --time-min=0-8:00:00
#SBATCH --exclude=node[001-030]
##SBATCH --partition=mcdermott
#SBATCH --array=0-99

source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10_filter_signalHPv00/*.hdf5'
# source_fn_regex="$SCRATCH_PATH"'/data_pitchnet/PND_synthetic/noise_UMNm_snr_neg10pos10_phase01_filter_signalLPv00/*.hdf5'

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE
echo $(hostname)

module add openmind/singularity

singularity exec --nv \
-B /home \
-B /om \
-B /om2 \
-B /om4 \
-B /nobackup \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python -u stimuli_compute_statistics.py \
-r "${source_fn_regex}" \
-d "SPECTRAL_STATISTICS_v00" \
-j $job_idx
