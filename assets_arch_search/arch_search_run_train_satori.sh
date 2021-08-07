job_idx=$(( $1 - 0 ))
SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(( $2 - 0 ))
PARALLEL_JOB_COUNTER=$(( $3 - 0 ))
PARALLEL_SLOT_NUMBER=$(( $4 - 0 ))

export PATH=$HOME/opt/bin:$PATH

echo "=============================="
echo "job_idx=$job_idx"
echo "HOSTNAME::SINGULARITYENV_CUDA_VISIBLE_DEVICES=$HOSTNAME::GPU-$SINGULARITYENV_CUDA_VISIBLE_DEVICES"
echo "PARALLEL_JOB_COUNTER=$PARALLEL_JOB_COUNTER"
echo "PARALLEL_SLOT_NUMBER=$PARALLEL_SLOT_NUMBER"
echo "DEBUG (slot::host::gpu)=$PARALLEL_SLOT_NUMBER::$HOSTNAME::$SINGULARITYENV_CUDA_VISIBLE_DEVICES"
echo "=============================="

DATA_TRAIN='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords'
DATA_EVAL='/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords'

SOURCE_CONFIG_FN='config_arch_search_v02.json'
OUTPUT_DIR_PATTERN="/saved_models/arch_search_v02/arch_{:04d}"
OUTPUT_LOG_FN=$(printf "/nobackup/users/msaddler/pitchnet/saved_models/arch_search_v02/logs_train/arch_%04d.log" ${job_idx})

echo $OUTPUT_LOG_FN
echo $(hostname)

if [[ "$(tail -1 $OUTPUT_LOG_FN)" == "Training stopped." ]]; then
    echo "[Training stopped.] $OUTPUT_LOG_FN" &> $(printf "satori_slurm_arch_%04d.out" ${job_idx})
    exit 1
else
    echo "$OUTPUT_LOG_FN" &> $(printf "satori_slurm_arch_%04d.out" ${job_idx})
fi

export SINGULARITYENV_CUDA_VISIBLE_DEVICES
singularity exec --nv \
-B /nobackup/users/msaddler/ \
-B /nobackup/users/msaddler/data_pitchnet:/data \
-B /nobackup/users/msaddler/pitchnet/saved_models:/saved_models \
-B /nobackup/users/msaddler/pitchnet/ibmHearingAid:/code_location \
docker://afrancl/ibm-hearing-aid-satori:tensorflow \
python -u /nobackup/users/msaddler/pitchnet/assets_archsearch/arch_search_run_train.py \
-o "${OUTPUT_DIR_PATTERN}" \
-c "${SOURCE_CONFIG_FN}" \
-j ${job_idx} \
-dt $DATA_TRAIN \
-de $DATA_EVAL \
2>&1 | tee $OUTPUT_LOG_FN
