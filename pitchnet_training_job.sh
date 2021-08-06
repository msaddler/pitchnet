#!/bin/bash
#
#SBATCH --job-name=pitchnet_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=6
#SBATCH --mem=18000
#SBATCH --gres=gpu:QUADRORTX6000:1
#SBATCH --time=1-24:00:00
#SBATCH --array=83,154,190,191,286,288,302,335,338,346
##SBATCH --exclude=node063
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
##SBATCH --dependency=afterok:17887282
#SBATCH --requeue

ZPJID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
OUTDIR='/saved_models/arch_search_v02_topN/DEMO/arch_'$ZPJID

OUTPUT_LOG_FN='/om2/user/msaddler/pitchnet'$OUTDIR'/output_train.log'
DATA_TRAIN='/nese/mit/group/mcdermott/msaddler/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[0-7]*.tfrecords'
DATA_EVAL='/nese/mit/group/mcdermott/msaddler/data_pitchnet/PND_v08/noise_TLAS_snr_neg10pos10/sr20000_cf100_species002_spont070_BW10eN1_IHC3000Hz_IHC7order/bez2018meanrates_0[8-9]*.tfrecords'


echo $OUTDIR
echo $OUTPUT_LOG_FN
echo $(hostname)

singularity exec --nv \
-B /home \
-B /nobackup \
-B /om \
-B /om2 \
-B /om4 \
-B /nese \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
python /om2/user/msaddler/pitchnet/assets_network/run_train_or_eval.py $OUTDIR \
-dt "$DATA_TRAIN" \
-de "$DATA_EVAL" \
-t -e -f \
2>&1 | tee $OUTPUT_LOG_FN
