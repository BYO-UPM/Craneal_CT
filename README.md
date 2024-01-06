# Development of an algorithm to segment the skull structure using Deep Learning

## Yichun Sun

## Tutor: Juan Ignacio Godino Llorente


This is the repository for the final Master thesis work. 

Universidad Politécnica de Madrid

Msc. of Science in Signal Theory and Communications

### Contents

CAT_scans_Original: Full CAT scans in DICOM files

CAT_scans_Preprocessed: Slices selected with manual masks in PNG files

Codes: 2D & 3D U-Nets, their variants (with backbone ResNet50, VGG16)

### Development Environment (important)

Python version: 3.10.12

For 2D architectures: TensorFlow > 2.14, Keras > 2.14, Segmentation-models 1.0.1

For 3D architectures: TensorFlow == 2.8, Keras == 2.8, Keras_applications 1.0.8, Segmentation-models-3D 1.0.8

After the updating in Dec. 2023, CUDA 12.2 doesn't support TensorFlow and Keras 2.8 anymore, the codes can run but GPU not working. There are two methods to train the 3D architectures with GPU in this case:

(1) Use `apt update && apt install cuda-11-8` to get previous CUDA version

(2) In Google Colab: Use Colab’s fallback runtime version by selecting "Use fallback runtime version" command when connecting to a runtime from the Command Palette.
