#!/bin/bash

OUTDIR=$1
TFRECORDS_REGEX=$2
EFN_PREFIX=$3
WRITE_PROBS_OUT=$4

echo "============"
echo "OUTDIR="$OUTDIR
echo "TFRECORDS_REGEX="$TFRECORDS_REGEX
echo "EFN_PREFIX="$EFN_PREFIX
echo "WRITE_PROBS_OUT="$WRITE_PROBS_OUT
echo "============"

# python pitchnet_evaluate_best.py \
# -de "/data/bernox2005/lowharm_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}lowharm_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/mooremoore2003/freqshifted_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}freqshifted_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/moore1985/mistunedharm_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}mistunedharm_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/oxenham2004/transposedtones_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}transposedtones_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/shackcarl1994/altphase_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}altphase_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/mcpherson2020/testsnr_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}testsnr_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/mcpherson2020/testspl_v01/$TFRECORDS_REGEX" \
# -efn "${EFN_PREFIX}testspl_v01_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo $WRITE_PROBS_OUT

# python pitchnet_evaluate_best.py \
# -de "/data/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords" \
# -efn "EVAL_validation_bestckpt.json" \
# -o "$OUTDIR" \
# -wpo 0
