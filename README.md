### Pitchnet

This repository contains code and models to accompany the [manuscript](https://doi.org/10.1101/2020.11.19.389999): "Deep neural network models reveal interplay of peripheral coding and stimulus statistics in pitch perception" by Mark R. Saddler, Ray Gonzalez, and Josh H. McDermott.

### System requirements

A Singularity environment ([tensorflow-1.13.1-pitchnet.simg](https://drive.google.com/file/d/1Dvvx5D9kIiHWhHeNg2D6_SyfTFSViQz2/view?usp=sharing)) with all required software is included in this code release. The Singularity environment was built on a linux-gnu operating system (CentOS Linux 7) with Singularity version 3.4.1. To run models using the Tensorflow version included in the Singularity environment, a CUDA Version: 11.2 - supported GPU is required (estimated runtimes reported in `DEMO.ipynb` are based on one NVIDIA titan-x GPU).

To run the Singularity environment and list included Python packages:
```
$ cd pitchnet
$ singularity exec --nv -B ./packages:/packages tensorflow-1.13.1-pitchnet.simg bash
$ pip list
    Package              Version
    -------------------- ----------------------
    -ip                  19.1
    absl-py              0.7.1
    astor                0.7.1
    attrs                19.1.0
    audioread            2.1.9
    backcall             0.1.0
    bleach               3.1.0
    cffi                 1.14.6
    cycler               0.10.0
    Cython               0.29.24
    dask                 2.6.0
    decorator            4.4.0
    defusedxml           0.6.0
    entrypoints          0.3
    enum34               1.1.6
    gast                 0.2.2
    grpcio               1.20.1
    h5py                 2.9.0
    ipykernel            5.1.0
    ipython              7.5.0
    ipython-genutils     0.2.0
    ipywidgets           7.4.2
    jedi                 0.13.3
    Jinja2               2.10.1
    joblib               0.14.1
    jsonschema           3.0.1
    jupyter              1.0.0
    jupyter-client       5.2.4
    jupyter-console      6.0.0
    jupyter-core         4.4.0
    jupyter-http-over-ws 0.0.6
    Keras-Applications   1.0.7
    Keras-Preprocessing  1.0.9
    kiwisolver           1.1.0
    librosa              0.7.2
    llvmlite             0.31.0
    Markdown             3.1
    MarkupSafe           1.1.1
    matplotlib           3.0.3
    mistune              0.8.4
    mock                 2.0.0
    nbconvert            5.5.0
    nbformat             4.4.0
    notebook             5.7.8
    numba                0.47.0
    numpy                1.16.3
    pandas               0.24.2
    pandocfilters        1.4.2
    parso                0.4.0
    pbr                  5.2.0
    pexpect              4.7.0
    pickleshare          0.7.5
    pip                  20.3.4
    prometheus-client    0.6.0
    prompt-toolkit       2.0.9
    protobuf             3.7.1
    ptyprocess           0.6.0
    pycparser            2.20
    pycurl               7.43.0
    Pygments             2.3.1
    pygobject            3.20.0
    pyparsing            2.4.0
    pyrsistent           0.15.1
    python-apt           1.1.0b1+ubuntu0.16.4.2
    python-dateutil      2.8.0
    pytz                 2021.1
    pyzmq                18.0.1
    qtconsole            4.4.3
    resampy              0.2.2
    scikit-learn         0.22.2.post1
    scipy                1.4.1
    Send2Trash           1.5.0
    setuptools           41.0.1
    six                  1.12.0
    SoundFile            0.10.3.post1
    tensorboard          1.13.1
    tensorflow-estimator 1.13.0
    tensorflow-gpu       1.13.1
    termcolor            1.1.0
    terminado            0.8.2
    testpath             0.4.2
    toolz                0.11.1
    tornado              6.0.2
    traitlets            4.3.2
    wcwidth              0.1.7
    webencodings         0.5.1
    Werkzeug             0.15.2
    wheel                0.29.0
    widgetsnbextension   3.4.2
```

### Installation instructions

All code except the Python wrapper around the Bruce et al. (2018) Auditory Nerve Model (`/packages/bez2018model`) comes compiled and can be run within the Singularity environment without additional installation. To install the auditory nerve model:
```
$ cd pitchnet
$ singularity exec --nv -B ./packages:/packages tensorflow-1.13.1-pitchnet.simg bash
$ cd /packages/bez2018model
$ python setup.py build_ext --inplace
```
This will compile the Cython code for the bez2018model Python wrapper and should take < 1 minute.


### DEMO

See the `DEMO.ipynb` Jupyter Notebook for a walk-through of how to generate simulated auditory nerve representations of example stimuli and evaluate our trained deep neural networks on them.


### Data and model availability

Sound waveform datasets and trained model checkpoints are available for [download from Google Drive](https://drive.google.com/drive/folders/1OhSzszxCnBfQ6cuJIaKuv5A_uoqks-H8?usp=sharing). To run code without modifying paths, downloaded models should be placed in the `pitchnet/models` directory. To keep the size of this code release manageable, we have only included trained model checkpoints for our `default` (most human-like) model (`models.zip`). The default model consists of 10 distinct deep neural network architectures (the 10 best-performing networks from our large-scale random DNN architecture search):
```
$ du -hd1 models/default/       
    24M     models/default/arch_0083
    105M    models/default/arch_0154
    82M     models/default/arch_0190
    59M     models/default/arch_0191
    2.4G    models/default/arch_0286
    115M    models/default/arch_0288
    217M    models/default/arch_0302
    120M    models/default/arch_0335
    117M    models/default/arch_0338
    107M    models/default/arch_0346
    3.4G    models/default/
```
Models optimized for alternative sound statistics can be obtained by training the above 10 architectures on alternative datasets provided in the Google Drive. Models optimized for alternative cochleae can be obtained by training the above 10 architectures on the main training dataset in conjunction with altered auditory nerve model parameters. In practice, the training of these models incurs substantial storage (for the simulated auditory nerve representations) and computing costs, so all trained model checkpoints are available upon request to the authors.

Sound waveform datasets (which can be converted into auditory nerve representations using the released code) are available for [download from Google Drive](https://drive.google.com/drive/folders/1OhSzszxCnBfQ6cuJIaKuv5A_uoqks-H8?usp=sharing):
* Pitchnet natural sounds training dataset: `dataset_pitchnet_train.tar.gz`
    - Stimuli 0000000-1680000 are used for training
    - Sitmuli 1680000-2100000 are held-out for validation
* Datasets for psychophysical experiments A-E: `dataset_psychophysics.tar.gz`
Simulated auditory nerve representations of all datasets used to train and evaluate models are also available upon request to the authors.
```
Main training + validation dataset (speech + instrument clips embedded in real-world background noise):
|__ Sound waveforms only: 120G
|__ bez2018model nervegrams: 810G per cochlear model variant

Psychophysical experiment datasets
|__ Expt. A: Bernstein & Oxenham (2005)
    |__ Sound waveforms only: 1.4G
    |__ bez2018model nervegrams: 49G per cochlear model variant
|__ Expt. B: Shackleton & Carlyon (1994)
    |__ Sound waveforms only: 678M
    |__ bez2018model nervegrams: 25G per cochlear model variant
|__ Expt. C: Moore & Moore (2003)
    |__ Sound waveforms only: 1.5G
    |__ bez2018model nervegrams: 57G per cochlear model variant
|__ Expt. D: Moore et al. (1985)
    |__ Sound waveforms only: 1.5G
    |__ bez2018model nervegrams: 56G per cochlear model variant
|__ Expt. E: Oxenham et al. (2004)
    |__ Sound waveforms only: 902M
    |__ bez2018model nervegrams: 17G per cochlear model variant
```

Please contact Mark R. Saddler (msaddler@mit.edu) with questions or requests for models / data / code.
