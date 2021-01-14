#!/bin/bash
#
#SBATCH --job-name=pitchnet_eval
#SBATCH --out="/om2/user/msaddler/pitchnet/slurm_pitchnet_eval-%A_%a.out"
##SBATCH --gres=gpu:tesla-v100:1
#SBATCH --gres=gpu:QUADRORTX6000:1
##SBATCH --gres=gpu:GEFORCEGTX1080TI:1
##SBATCH --gres=gpu:GEFORCERTX2080TI:1
#SBATCH --mem=18000
#SBATCH --cpus-per-task=6
#SBATCH --time=0-12:00:00
##SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --array=0-9
##SBATCH --array=83,154,190,191,286,288,302,335,338,346

# ZPJID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
# OUTDIR='/saved_models/arch_search_v02_topN/PND_v08_noise_TLAS_snr_posInf/arch_'$ZPJID
OUTDIR='/om2/user/msaddler/pitchnet/saved_models/models_metamer/arch_0302_PND_v08_TLAS_snr_neg10pos10_AN_RSB_noise0000_classification0'
DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"

PATH_DATA="/om/user/msaddler/data_pitchnet"
# PATH_DATA="$SCRATCH_PATH/data_pitchnet"
PATH_SAVED_MODELS="/om2/user/msaddler/pitchnet/saved_models"
PATH_CODE_LOCATION="/om2/user/msaddler/pitchnet/ibmHearingAid"
PATH_PYTHON_PACKAGES="/om2/user/msaddler/python-packages"

TFRECORDS_REGEX="$DATA_TAG/*.tfrecords"
# TFRECORDS_REGEX="$DATA_TAG/bez2018meanrates_0[8-9]*.tfrecords"
EFN_PREFIX='EVAL_SOFTMAX_'
OUTPUT_LOG_FN=$OUTDIR'/output_eval.log'

echo "[START EVALUATION] $OUTPUT_LOG_FN"
singularity exec --nv \
-B /om2 \
-B $PATH_DATA:/data \
-B $PATH_SAVED_MODELS:/saved_models \
-B $PATH_CODE_LOCATION:/code_location \
-B $PATH_PYTHON_PACKAGES:/python-packages \
/om2/user/msaddler/singularity-images/tfv1.13_unet.simg \
./pitchnet_run_eval.sh $OUTDIR $TFRECORDS_REGEX $EFN_PREFIX $OUTPUT_LOG_FN
echo "[END EVALUATION] $OUTPUT_LOG_FN"
