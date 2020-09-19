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

## CHOOSE THE OUTPUT DIRECTORY FROM A HARD-CODED LIST
declare -a outdir_list=(
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed0"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed1"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed2"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed3"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed4"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed5"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed6"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed7"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed8"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed9"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed0"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed1"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed2"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed3"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed4"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed5"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed6"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed7"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed8"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191_seed9"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed0"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed1"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed2"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed3"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed4"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed5"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed6"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed7"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed8"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191_seed9"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed0"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed1"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed2"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed3"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed4"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed5"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed6"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed7"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed8"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191_seed9"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0083"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0154"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0190"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0191"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0286"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0288"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0302"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0335"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0338"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/arch_0346"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0083"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0154"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0190"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0191"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0286"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0288"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0302"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0335"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0338"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order/arch_0346"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0083"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0154"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0190"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0191"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0286"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0288"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0302"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0335"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0338"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order/arch_0346"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0083"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0154"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0190"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0191"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0286"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0288"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0302"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0335"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0338"
    "/saved_models/arch_search_v02_topN/sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order/arch_0346"
)
OUTDIR=${outdir_list[$job_idx]}

## CHOOSE THE DATA_TAG BASED ON THE OUTPUT DIRECTORY
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont1eN1_BW10eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW05eN1_IHC3000Hz_IHC7order"
fi
if [[ "$OUTDIR" == *"sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order"* ]]; then
  DATA_TAG="sr20000_cf100_species002_spont070_BW20eN1_IHC3000Hz_IHC7order"
fi

# DATA_TRAIN='/data/PND_v08/noise_TLAS_snr_neg10pos10/'$DATA_TAG'/bez2018meanrates_0[0-7]*.tfrecords'
# DATA_EVAL='/data/PND_v08/noise_TLAS_snr_neg10pos10/'$DATA_TAG'/bez2018meanrates_0[8-9]*.tfrecords'
# OUTPUT_LOG_FN=$OUTDIR'/output_train.log'

# echo "[START TRAINING] $OUTPUT_LOG_FN" &> $(printf "slurm_run_train_satori-%04d.out" ${job_idx})
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES
# singularity exec --nv \
# -B $PATH_DATA:/data \
# -B $PATH_SAVED_MODELS:/saved_models \
# -B $PATH_CODE_LOCATION:/code_location \
# docker://afrancl/ibm-hearing-aid-satori:tensorflow \
# ./pitchnet_run_train.sh $OUTDIR $DATA_TRAIN $DATA_EVAL $OUTPUT_LOG_FN
# echo "[END TRAINING] $OUTPUT_LOG_FN" >> $(printf "slurm_run_train_satori-%04d.out" ${job_idx})


# OUTDIR=$(printf "/saved_models/arch_search_v02/arch_%04d" ${job_idx})
TFRECORDS_REGEX="$DATA_TAG/*.tfrecords"
EFN_PREFIX='EVAL_SOFTMAX_'
OUTPUT_LOG_FN=$OUTDIR'/output_eval.log'

echo "[START EVALUATION] $OUTPUT_LOG_FN" &> $(printf "slurm_run_eval_satori-%04d.out" ${job_idx})
export SINGULARITYENV_CUDA_VISIBLE_DEVICES
singularity exec --nv \
-B $PATH_DATA:/data \
-B $PATH_SAVED_MODELS:/saved_models \
-B $PATH_CODE_LOCATION:/code_location \
docker://afrancl/ibm-hearing-aid-satori:tensorflow \
./pitchnet_run_eval.sh $OUTDIR $TFRECORDS_REGEX $EFN_PREFIX $OUTPUT_LOG_FN
echo "[END EVALUATION] $OUTPUT_LOG_FN" >> $(printf "slurm_run_eval_satori-%04d.out" ${job_idx})
