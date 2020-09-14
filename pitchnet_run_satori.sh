job_idx=$(( $1 - 0 ))
CUDA_VISIBLE_DEVICES=$(( $2 - 0 ))
PARALLEL_JOB_COUNTER=$(( $3 - 0 ))
PARALLEL_SLOT_NUMBER=$(( $4 - 0 ))

export PATH=$HOME/opt/bin:$PATH

echo "=============================="
echo "job_idx=$job_idx"
echo "HOSTNAME::CUDA_VISIBLE_DEVICES=$HOSTNAME::GPU-$CUDA_VISIBLE_DEVICES"
echo "PARALLEL_JOB_COUNTER=$PARALLEL_JOB_COUNTER"
echo "PARALLEL_SLOT_NUMBER=$PARALLEL_SLOT_NUMBER"
echo "=============================="

DATA_PATH="/nobackup/users/msaddler/data_pitchnet"
SAVED_MODELS_PATH="/nobackup/users/msaddler/pitchnet/saved_models"
CODE_LOCATION_PATH="/nobackup/users/msaddler/pitchnet/ibmHearingAid"

OUTDIR=$(printf "/saved_models/arch_search_v02/arch_%04d" ${job_idx})
TFRECORDS_REGEX='sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/*.tfrecords'
EFN_PREFIX='EVAL_SOFTMAX_'
WRITE_PROBS_OUT=1

# singularity exec --nv \
# -B $DATA_PATH:/data \
# -B $SAVED_MODELS_PATH:/saved_models \
# -B $CODE_LOCATION_PATH:/code_location \
# docker://afrancl/ibm-hearing-aid-satori:tensorflow \
# ./pitchnet_run_eval.sh $OUTDIR $TFRECORDS_REGEX $EFN_PREFIX $WRITE_PROBS_OUT
