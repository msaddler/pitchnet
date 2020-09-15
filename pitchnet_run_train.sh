OUTDIR=$1
DATA_TRAIN=$2
DATA_EVAL=$3
OUTPUT_LOG_FN=$4

echo "------------------------------------------------" > $OUTPUT_LOG_FN
echo "| (host::gpu)=$(hostname)::GPU-$CUDA_VISIBLE_DEVICES" >> $OUTPUT_LOG_FN
echo "| OUTDIR=$OUTDIR" >> $OUTPUT_LOG_FN
echo "| DATA_TRAIN=$DATA_TRAIN" >> $OUTPUT_LOG_FN
echo "| DATA_EVAL=$DATA_EVAL" >> $OUTPUT_LOG_FN
echo "| OUTPUT_LOG_FN=$OUTPUT_LOG_FN" >> $OUTPUT_LOG_FN
echo "------------------------------------------------" >> $OUTPUT_LOG_FN

python /code_location/multi_gpu/run_train_or_eval.py "$OUTDIR" \
-dt "$DATA_TRAIN" \
-de "$DATA_EVAL" \
-t -e -f \
>> "$OUTPUT_LOG_FN" 2>&1
