### Pitchnet

This repository contains code and models to accompany our [paper](https://doi.org/10.1101/2020.11.19.389999): "Deep neural network models reveal interplay of peripheral coding and stimulus statistics in pitch perception" by Mark R. Saddler, Ray Gonzalez, and Josh H. McDermott.


### System requirements

A [Singularity](https://sylabs.io/guides/3.0/user-guide/index.html) environment ([tensorflow-1.13.1-pitchnet.simg](https://drive.google.com/file/d/1Dvvx5D9kIiHWhHeNg2D6_SyfTFSViQz2/view?usp=sharing)) with all required software is included in this code release. The Singularity environment was built on a linux-gnu operating system (CentOS Linux 7) with Singularity version 3.4.1. To run models using the Tensorflow version included in the Singularity environment, a CUDA Version: 11.2 - supported GPU is required (estimated runtimes reported in [DEMO.ipynb](DEMO.ipynb) are based on one NVIDIA titan-x GPU).

To open a bash shell in the Singularity environemnt:
```
$ singularity exec --nv -B ./packages:/packages tensorflow-1.13.1-pitchnet.simg bash
```


### Installation instructions

Clone this repository: `git clone --recurse-submodules git@github.com:msaddler/pitchnet.git`

All code except the Python wrapper around the Bruce et al. (2018) Auditory Nerve Model (`/packages/bez2018model`) comes compiled and can be run within the Singularity environment without additional installation. To install the auditory nerve model:
```
$ cd pitchnet
$ singularity exec --nv -B ./packages:/packages tensorflow-1.13.1-pitchnet.simg bash
$ cd /packages/bez2018model
$ python setup.py build_ext --inplace
```
This will compile the Cython code for the bez2018model Python wrapper and should take < 1 minute.


### DEMO

See the [DEMO.ipynb](DEMO.ipynb) Jupyter Notebook for a walk-through of how to generate simulated auditory nerve representations of example stimuli and evaluate our trained deep neural networks on them.


### Data and model availability

Sound waveform datasets and trained model checkpoints are available for [download from Google Drive](https://drive.google.com/drive/folders/1OhSzszxCnBfQ6cuJIaKuv5A_uoqks-H8?usp=sharing). To run code without modifying paths, downloaded models should be placed in the `pitchnet/models` directory. To keep the size of this code release manageable, we have only included trained model checkpoints for our `default` (most human-like) model. The default model consists of 10 distinct deep neural network architectures (the 10 best-performing networks from our large-scale random DNN architecture search):
```
$ du -hd1 models/default/       
    2.2G	default/arch_0083
    2.2G	default/arch_0154
    2.2G	default/arch_0190
    2.2G	default/arch_0191
    4.5G	default/arch_0286
    2.2G	default/arch_0288
    2.3G	default/arch_0302
    2.2G	default/arch_0335
    2.2G	default/arch_0338
    2.2G	default/arch_0346
    25G	    default/
```
Models optimized for alternative sound statistics can be obtained by training the above 10 architectures on alternative datasets provided in the Google Drive. Models optimized for alternative cochleae can be obtained by training the above 10 architectures on the main training dataset in conjunction with altered auditory nerve model parameters. In practice, the training of these models incurs substantial storage (for the simulated auditory nerve representations) and computing costs, so all trained model checkpoints are available upon request to the authors.

Sound waveform datasets (which can be converted into auditory nerve representations using the released code) are available for [download from Google Drive](https://drive.google.com/drive/folders/1OhSzszxCnBfQ6cuJIaKuv5A_uoqks-H8?usp=sharing):
* Pitchnet natural sounds training dataset: `dataset_pitchnet_train.tar.gz`
    - Stimuli 0000000-1680000 are used for training
    - Sitmuli 1680000-2100000 are held-out for validation
* Datasets for psychophysical experiments A-E: `dataset_psychophysics.tar.gz`

Simulated auditory nerve representations (with our default / most human-like parameters) of all datasets used to train and evaluate models are also available on the [Google Drive](https://drive.google.com/drive/folders/1OhSzszxCnBfQ6cuJIaKuv5A_uoqks-H8?usp=sharing). Simulated auditory nerve representations with altered parameters are available upon request to the authors.
```
Main training + validation dataset (speech + instrument clips embedded in real-world background noise):
|__ Sound waveforms only: 120G
|__ bez2018model nervegrams: 810G per cochlear model variant

Psychophysical experiment datasets
|__ Expt. A: Bernstein & Oxenham (2005)
    |__ Sound waveforms only: 1.4G
    |__ bez2018model nervegrams: 22G per cochlear model variant
|__ Expt. B: Shackleton & Carlyon (1994)
    |__ Sound waveforms only: 678M
    |__ bez2018model nervegrams: 11G per cochlear model variant
|__ Expt. C: Moore & Moore (2003)
    |__ Sound waveforms only: 1.5G
    |__ bez2018model nervegrams: 26G per cochlear model variant
|__ Expt. D: Moore et al. (1985)
    |__ Sound waveforms only: 1.5G
    |__ bez2018model nervegrams: 25G per cochlear model variant
|__ Expt. E: Oxenham et al. (2004)
    |__ Sound waveforms only: 902M
    |__ bez2018model nervegrams: 7.3G per cochlear model variant
```

Please contact Mark R. Saddler (msaddler@mit.edu) with questions or requests for models / data / code.
