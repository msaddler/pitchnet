#!/bin/bash
#
#SBATCH --job-name=synthetic_tones
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
##SBATCH --time-min=0-3:00:00
#SBATCH --exclude=node[001-030]
#SBATCH --array=0-99
##SBATCH --partition=mcdermott

dest_filename="$SCRATCH_PATH/data_pitchnet/PND_mfcc/PNDv08matched12_TLASmatched12_snr_neg10pos10_phase3/stim.hdf5"
num_parallel_jobs=100
num_total_stimuli=2100000
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

export HDF5_USE_FILE_LOCKING=FALSE

module add openmind/singularity

singularity exec \
-B /home \
-B /om \
-B /om2 \
-B /om4 \
-B /nobackup \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
python -u stimuli_generate_random_synthetic_tones.py \
-d "${dest_filename}" \
-j ${job_idx} \
-npj ${num_parallel_jobs} \
-nts ${num_total_stimuli} \
-isf 0
