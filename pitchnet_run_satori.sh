#!/bin/bash

job_idx=$(( $1 - 0 ))
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(( $2 - 0 ))
PARALLEL_JOB_COUNTER=$(( $3 - 0 ))
PARALLEL_SLOT_NUMBER=$(( $4 - 0 ))

export PATH=$HOME/opt/bin:$PATH

echo "-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"
echo "|. job_idx=$job_idx"
echo ".| (slot::host::gpu)=$PARALLEL_SLOT_NUMBER::$HOSTNAME::GPU-$SINGULARITYENV_CUDA_VISIBLE_DEVICES"
echo "|. PARALLEL_JOB_COUNTER=$PARALLEL_JOB_COUNTER"
echo "-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"

PATH_DATA="/nobackup/users/msaddler/data_pitchnet"
PATH_SAVED_MODELS="/nobackup/users/msaddler/pitchnet/saved_models"
PATH_CODE_LOCATION="/nobackup/users/msaddler/pitchnet/ibmHearingAid"
PATH_PYTHON_PACKAGES="/nobackup/users/msaddler/python-packages"

## CHOOSE THE OUTPUT DIRECTORY FROM A HARD-CODED LIST
declare -a outdir_list=(
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0083"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0154"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0190"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0191"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0286"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0288"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0302"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0335"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0338"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz_matched_tfcochlearn/arch_0346"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0083"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0154"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0190"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0191"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0286"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0288"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0302"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0335"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0338"
    "/saved_models/arch_search_v02_topN/connear_IHC4000Hz/arch_0346"
)
OUTDIR=${outdir_list[$job_idx]}

## CHOOSE THE DATA_TAG BASED ON THE OUTPUT DIRECTORY
DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW02eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW02eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW40eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW40eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW10eN1_IHC0320Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC0320Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW10eN1_IHC0250Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC0250Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr2000_cf1000_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"* ]]; then
  DATA_TAG="sr2000_cf1000_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr2000_cfI100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"* ]]; then
  DATA_TAG="sr2000_cfI100_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr2000_cfI250_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"* ]]; then
  DATA_TAG="sr2000_cfI250_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr2000_cfI500_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"* ]]; then
  DATA_TAG="sr2000_cfI500_species002_spont070_BW10eN1_IHC0050Hz_IHC7order"
fi

DATA_TRAIN='/data/PND_v08/noise_TLAS_snr_neg10pos10/'$DATA_TAG'/bez2018meanrates_0[0-7]*.tfrecords'
DATA_EVAL='/data/PND_v08/noise_TLAS_snr_neg10pos10/'$DATA_TAG'/bez2018meanrates_0[8-9]*.tfrecords'
OUTPUT_LOG_FN=$OUTDIR'/output_train.log'

# ## CHOOSE THE DATA_TAG BASED ON THE OUTPUT DIRECTORY
# if [[ "$OUTDIR" == *"noise_TLAS_snr_posInf"* ]]; then
#   DATA_TAG="noise_TLAS_snr_posInf"
# fi
# if [[ "$OUTDIR" == *"noise_TLAS_snr_pos10pos30"* ]]; then
#   DATA_TAG="noise_TLAS_snr_pos10pos30"
# fi
# if [[ "$OUTDIR" == *"noise_TLAS_snr_neg10pos10_filter_signalLPv01"* ]]; then
#   DATA_TAG="noise_TLAS_snr_neg10pos10_filter_signalLPv01"
# fi
# if [[ "$OUTDIR" == *"noise_TLAS_snr_neg10pos10_filter_signalHPv00"* ]]; then
#   DATA_TAG="noise_TLAS_snr_neg10pos10_filter_signalHPv00"
# fi

# # DATA_TRAIN='/data/PND_v08/'$DATA_TAG'/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords'
# # DATA_EVAL='/data/PND_v08/'$DATA_TAG'/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords'
# # OUTPUT_LOG_FN=$OUTDIR'/output_train.log'

# echo "[START TRAINING] $OUTPUT_LOG_FN" &> $(printf "slurm_run_train_satori-%04d.out" ${job_idx})
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES
# singularity exec --nv \
# -B $PATH_DATA:/data \
# -B $PATH_SAVED_MODELS:/saved_models \
# -B $PATH_CODE_LOCATION:/code_location \
# -B $PATH_PYTHON_PACKAGES:/python-packages \
# docker://afrancl/ibm-hearing-aid-satori:tensorflow \
# ./pitchnet_run_train.sh $OUTDIR $DATA_TRAIN $DATA_EVAL $OUTPUT_LOG_FN
# echo "[END TRAINING] $OUTPUT_LOG_FN" >> $(printf "slurm_run_train_satori-%04d.out" ${job_idx})


# OUTDIR=$(printf "/saved_models/arch_search_v02/arch_%04d" ${job_idx})
# DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"
TFRECORDS_REGEX="$DATA_TAG/*.tfrecords"
EFN_PREFIX='EVAL_SOFTMAX_'
OUTPUT_LOG_FN=$OUTDIR'/output_eval.log'

echo "[START EVALUATION] $OUTPUT_LOG_FN" &> $(printf "slurm_run_eval_satori-%04d.out" ${job_idx})
export SINGULARITYENV_CUDA_VISIBLE_DEVICES
singularity exec --nv \
-B $PATH_DATA:/data \
-B $PATH_SAVED_MODELS:/saved_models \
-B $PATH_CODE_LOCATION:/code_location \
-B $PATH_PYTHON_PACKAGES:/python-packages \
docker://afrancl/ibm-hearing-aid-satori:tensorflow \
./pitchnet_run_eval.sh $OUTDIR $TFRECORDS_REGEX $EFN_PREFIX $OUTPUT_LOG_FN
echo "[END EVALUATION] $OUTPUT_LOG_FN" >> $(printf "slurm_run_eval_satori-%04d.out" ${job_idx})
