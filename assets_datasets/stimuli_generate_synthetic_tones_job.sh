#!/bin/bash
#
#SBATCH --partition=mcdermott
#SBATCH --job-name=synth_tones
#SBATCH --out="trash/slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2500
#SBATCH --nodes=1
#SBATCH --time=0-3:30:00
#SBATCH --array=0-99
##SBATCH --dependency=afterok:12702768

offset=100
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
hdf5_filename='/om/user/msaddler/data_pitchnet/bernox2005/SyntheticTonesBandpass/SyntheticTonesBandpass_v0.hdf5-'

# python -u stimuli_generate_synthetic_tones.py "${hdf5_filename}${job_idx}" 6000

python -u stimuli_jwss_background_noise.py "${hdf5_filename}${job_idx}" "/om/user/msaddler/data_pitchnet/JarrodWiktorSoundSegments.hdf5"
