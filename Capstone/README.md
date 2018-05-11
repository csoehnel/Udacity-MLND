# Dense Disparity Estimation from Monocular Image Cues

Machine Learning Engineer Nanodegree
Capstone Project Report

Christoph Soehnel

## Files

* datahandling.py: This file contains functions for serving data.
* evaluation.py: This file contains the evaluation logic.
* model_discriminator.py: This file defines the discriminator model.
* model_generator.py: This file defines the generator model as a U-Net.
* python.pfm: This file contains the function to read/write the disparity files in PFM format.
* test.py: This is the main file for testing.
* train.py: This is the main file for training.
* vislog.py: This file contains functions logging the training process.

## Prerequisites

The following set-up has been used for training and testing:

* NVIDIA GPU
* 12 GB Free system RAM

* Ubuntu 18.04 LTS
* Docker 17.12.1-ce + nvidia-docker2

* Python 3.5.2
* Keras 2.1.6
* Pillow 5.1.0
* h5py 2.7.1
* numpy 1.14.2
* opencv-python 3.4.0.12
* pydot 1.0.29
* requests 2.18.4
* tensorflow 1.8.0

###Image data:
https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass_webp.tar

###Disparity data:
https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2

## Installing

1) Download and extract project files
2) Install prerequisites
3) Download and extract image data and disparity data

## Training

1) Edit train.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location
4) Start training with "python3 train.py"

## Testing

### For testing the final model described in the report
1) Edit test.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location
4) Start training with "python3 test.py"

### For testing another model
1) Edit test.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location
4) In dict "params" change path to actual location for key "path_generator"
5) Start training with "python3 test.py"

## Git

The project also resides on Github:
https://github.com/csoehnel/Udacity-MLND