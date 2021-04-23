#!/bin/bash

OUTDIR=$1
TFRECORDS_REGEX=$2
EFN_PREFIX=$3
OUTPUT_LOG_FN=$4

echo "------------------------------------------------" > $OUTPUT_LOG_FN
echo "| (host::gpu)=$(hostname)::GPU-$CUDA_VISIBLE_DEVICES" >> $OUTPUT_LOG_FN
echo "| OUTDIR=$OUTDIR" >> $OUTPUT_LOG_FN
echo "| TFRECORDS_REGEX=$TFRECORDS_REGEX" >> $OUTPUT_LOG_FN
echo "| EFN_PREFIX=$EFN_PREFIX" >> $OUTPUT_LOG_FN
echo "------------------------------------------------" >> $OUTPUT_LOG_FN


# python pitchnet_evaluate_best.py \
# -de "/data/bernox2005/lowharm_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}lowharm_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/mooremoore2003/freqshifted_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}freqshifted_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/moore1985/mistunedharm_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mistunedharm_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/oxenham2004/transposedtones_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}transposedtones_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/shackcarl1994/altphase_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}altphase_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/bernox2005/lowharm_v01_dbspl85/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}lowharm_v01_dbspl85_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/mooremoore2003/freqshifted_v01_dbspl85/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}freqshifted_v01_dbspl85_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/moore1985/mistunedharm_v01_dbspl85/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mistunedharm_v01_dbspl85_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/oxenham2004/transposedtones_v01_dbspl85/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}transposedtones_v01_dbspl85_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/shackcarl1994/altphase_v01_dbspl85/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}altphase_v01_dbspl85_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

python pitchnet_evaluate_best.py \
-de "/data/bernox2005/puretone_v00/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/*tfrecords" \
-efn "${EFN_PREFIX}_cohc1_puretone_v00_bestckpt.json" \
-o "$OUTDIR" \
-wpo 1 \
>> "$OUTPUT_LOG_FN" 2>&1

python pitchnet_evaluate_best.py \
-de "/data/bernox2005/puretone_v00/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order_cohc0/*tfrecords" \
-efn "${EFN_PREFIX}_cohc0_puretone_v00_bestckpt.json" \
-o "$OUTDIR" \
-wpo 1 \
>> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/mcpherson2020/testsnr_v01/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}testsnr_v01_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/mcpherson2020/testspl_v01/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}testspl_v01_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords" \
# # -efn "EVAL_validation_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 0 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/mcpherson2020/testsnr_v02/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}testsnr_v02_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/mcpherson2020/testspl_v02/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}testspl_v02_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/bernox2005/lowharm_v02/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}lowharm_v02_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/bernox2005/lowharm_v03/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}lowharm_v03_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# # python pitchnet_evaluate_best.py \
# # -de "/data/bernox2005/lowharm_v04/$TFRECORDS_REGEX" \
# # -efn "${EFN_PREFIX}lowharm_v04_bestckpt.json" \
# # -o "$OUTDIR" \
# # -wpo 1 \
# # >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/mcpherson2020/testspl_v03/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}testspl_v03_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/bernox2005/exact_v00/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}exact_v00_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/bernox2005/exact_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}exact_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/oxenham2004/transposedtones_v02/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}transposedtones_v02_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 1 \
# >> "$OUTPUT_LOG_FN" 2>&1

# python pitchnet_evaluate_best.py \
# -de "/data/PND_v08/noise_TLAS_snr_neg10pos10/$TFRECORDS_REGEX" \
# -efn "EVAL_validation_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 0 \
# >> "$OUTPUT_LOG_FN" 2>&1
