# Dense Disparity Estimation from Monocular Image Cues

### Machine Learning Engineer Nanodegree - Capstone Project

## Files/Folders

### logs/201804-19_09-28-42/console_test.log

This file contains the console output of the final model's testing output.

### logs/201804-19_09-28-42/discriminator.json

This file contains the actual discriminator configuration used for the the final model.

### logs/201804-19_09-28-42/discriminator.h5

The trained discriminator model.

### logs/201804-19_09-28-42/epoch0001-batch201115.jpg - epoch0305-batch20115.jpg

These files visualize the current generator performance after every epoch. One file consists of three vertically stacked images. From top to bottom: input image, ground truth disparity map, generated disparity map after current epoch.

### logs/201804-19_09-28-42/generator.h5

The trained generator model.

### logs/201804-19_09-28-42/generator.json

This file contains the actual generator configuration used for the final model.

### logs/201804-19_09-28-42/params_test.log

The parameters used for testing the model.

### logs/201804-19_09-28-42/params_train.log

The parameters used for training the model.

### proposal/proposal.pdf

The proposal for the project as pdf document.

### report/report.pdf

The report for the project as pdf document.

### datahandling.py

This file contains all the functions for managing the datasets.

### evaluation.py

This file contains the evaluation logic.

### model_discriminator.py

This file defines the discriminator model of the GAN.

### model_generator.py

This file defines the generator model of the GAN as a U-Net.

### python_pfm.py

This file contains the functions to read/write the disparity files in PFM format.

### test.py

This is the main file for testing.

### train.py

This is the main file for training.

### vislog.py

This file contains functions for logging and visualizing the training process.

## Prerequisites

The following set-up has been used for training and testing.

### Hardware

* AMD Ryzen 1700X 8-Core 3400 MHz
* 32 GB RAM DDR4-2666 MHz
* Samsung 960 Pro 512 GB M.2 SSD
* NVIDIA GeForce GTX 1080Ti 11 GB

### Software

* Ubuntu 18.04 LTS
* Docker 17.12.1-ce
* NVIDIA Docker 2.0.3
* NVIDIA Driver 390.48
* Python 3.5.2
* numpy 1.14.2
* Keras 2.1.6
* tensorflow 1.8.0
* Pillow 5.1.0
* opencv-python 3.4.0.12
* h5py 2.7.1
* requests 2.18.4
* pydot 1.0.29

### Image data

https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass_webp.tar

### Disparity data

https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2

## Installing

1) Clone or download and extract project files
2) Install prerequisites
3) Download and extract image data and disparity data

## Training

1) Edit train.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location of dataset
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location of dataset
4) Start training with "python3 train.py"

## Testing

### For testing the final model described in the report

1) Edit test.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location of dataset
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location of dataset
4) Start testing with "python3 test.py"

### For testing another model

1) Edit test.py
2) In dict "params" change substring "/home/XX/" for key "pathpattern_img" to actual location of dataset
3) In dict "params" change substring "/home/XX/" for key "pathpattern_dsp" to actual location of dataset
4) In dict "params" change path to actual location of trained generator model for key "path_generator"
5) Start testing with "python3 test.py"

## Author

Christoph Soehnel

## Git

The project also resides on Github:
https://github.com/csoehnel/Udacity-MLND
