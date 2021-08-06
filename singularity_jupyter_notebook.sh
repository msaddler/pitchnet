singularity exec --nv \
-B /home \
-B /nobackup \
-B /nese \
-B /om \
-B /om2 \
-B /om4 \
-B /om2/user/msaddler/python-packages:/python-packages \
-B /om/user/msaddler/data_pitchnet:/data \
-B /om2/user/msaddler/pitchnet/saved_models:/saved_models \
-B /om2/user/msaddler/pitchnet/packages:/packages \
/om2/user/msaddler/vagrant/tensorflow-1.13.1-pitchnet.simg \
/om2/user/msaddler/jupyter_notebook_job.sh
