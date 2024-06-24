# Data processing

## Data preprocessing

All DICOM files were converted to 2D PNG images (one for each slice). In addition, all CT scans were standardised with the same window size (500, 2000). Subsequently, images were normalised and equalised.

This preprocessing process is summarised in the foloowing image:

![prepro](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/datapreprocess.png)

## Data augmentation

We applied 5 types of data augmentation techniques to expand the dataset, they were:

- Intensity operations
  
  Change the intensity information of the original CT images with six new windows

<div align="center">
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/new_window.png" width="60%" alt="New Windows">
</div>
  
- Geometric transformations

  Horizontal and vertical flipping
  
  15% scaling down the x-axis

  Random rotation between ±180°
  
- Noise injection

  20% probability of randomly adding one type of noise, namely: Gaussian noise (mean=0, standard deviation=0.1), Salt & Pepper noise (salt probability=pepper probability=0.1), Laplace noise (mean=0, standard deviation=0.1), or Poisson noise (determined by pixel values).
  
- Cropping

  Circle crop

  Random crop (see the following example)
  <p align="center">
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/crop1.png" alt="Image 1" width="150" />
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/crop2.png" alt="Image 2" width="150" />
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/crop3.png" alt="Image 3" width="150" />
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/crop4.png" alt="Image 4" width="150" />
  </p>
  
- Filtering (Gaussian blur)

  Gaussian blur with kernel size = (15, 15) and sigma = (2.0, 3.0)
