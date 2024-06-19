# Automatic semantic segmentation of the osseous structures of the paranasal sinuses

xxxxx

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
