# Automatic semantic segmentation of the osseous structures of the paranasal sinuses
This repository contains the implementation of the automatic semantic segmentation model for the osseous structures of the paranasal sinuses, as described in our [paper](https://github.com/BYO-UPM/Craneal_CT). The model aims to assist in robotic-assisted surgeries by accurately delimiting critical anatomical structures, using U-Net based architectures enhanced with semi-supervised learning techniques.

<div align="center">
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Figures/example.gif" width="50%" alt="Segmentation Example">
</div>

## Overview

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.8+
- NumPy
- Other dependencies are listed in the `requirements.txt`

### Setup
To set up the software, run the following commands:
bash
git clone https://github.com/BYO-UPM/Craneal_CT
cd Craneal_CT
pip install -r requirements.txt


## Citation
If you use our model or dataset in your research, please cite our paper:
@article{yichun2024automatic,
  title={Automatic Semantic Segmentation of the Osseous Structures of the Paranasal Sinuses},
  author={Yichun Sun, Alejandro Guerrero-López, Julián D. Arias-Londoño, Juan I. Godino-Llorente},
  journal={xxx},
  year={2024},
  volume={XX},
  number={XX},
  pages={xx-xx}
}


### Contents

CAT_scans_Original: Full CAT scans in DICOM files

CAT_scans_Preprocessed: Slices selected with manual masks in PNG files

Codes: 2D & 3D U-Nets, their variants (with backbone ResNet50 and VGG16)

### Development Environment

Python version: 3.10.13

For 2D architectures: PyTorch 2.2.1, Segmentation-models-pytorch 0.3.3

For 3D architectures: TensorFlow == 2.8, Keras == 2.8, Keras_applications 1.0.8, Segmentation-models-3D 1.0.8

After the updating in Dec. 2023, CUDA 12.2 doesn't support TensorFlow and Keras 2.8 anymore, the codes can run but GPU not working. There are two methods to train the 3D architectures with GPU in this case:

(1) Use `apt update && apt install cuda-11-8` to get previous CUDA version

(2) In Google Colab: Use Colab’s fallback runtime version by selecting "Use fallback runtime version" command when connecting to a runtime from the Command Palette.
