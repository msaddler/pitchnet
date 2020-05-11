#!/bin/bash
#
#SBATCH --job-name=tfrecords_from_hdf5
#SBATCH --out="trash/tfrecords_slurm-%A_%a.out"
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500
#SBATCH --nodes=1
#SBATCH --time=0-2:30:00
#SBATCH --array=0-599
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
##SBATCH --dependency=afterok:14990544
#SBATCH --exclude=node[001-030]

### Define source_regex and dest_filename here (use single quotes to prevent regex from expanding)
source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/sr2000_cf1000_species002_spont070_BW10eN1_IHC0050Hz_IHC7order/bez2018meanrates*.hdf5'
# source_regex="$SCRATCH_PATH"'/data_pitchnet/PND_synthetic/noise_UMNm_snr_neg10pos10_phase01_filter_signalBPv00/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates*.hdf5'
# source_regex='/om/user/msaddler/data_pitchnet/bernox2005/FixedFilter_f0min100_f0max300/sr20000_cf100_species002_spont070_BW30eN1_IHC3000Hz_IHC7order/bez2018meanrates*.hdf5'
# source_regex='/om/user/msaddler/data_pitchnet/bernox2006/nharm10_f0min100_f0max300_TENlevel*/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/bez2018meanrates*.hdf5'

jobs_per_source_file=1
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

module add openmind/singularity

singularity exec --nv \
-B /om \
-B /om2 \
-B /nobackup \
/om2/user/msaddler/singularity-images/tensorflow-1.13.1-gpu-py3.img \
python -u tfrecords_from_hdf5.py "${source_regex}" ${job_idx} ${jobs_per_source_file}
