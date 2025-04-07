# Available Models

Due to the file size limitation in GitHub, please view our available models at this [link](https://huggingface.co/adayc/skull_segmentation).

## Final Model

The final model is stored in folder `Final_Model`.

The final model was obtained by training the true labels of Subsets 1 & 2, and all the pseudo-labels of Subsets 2 & 3. It can be used for further **transfer learning** applications. DO NOT use this model for testing on our datasets, as the training set already contained all CT slices in our datasets.

## 2D Models

You can use these models for **reproduction**. See our [paper](https://doi.org/10.1016/j.compmedimag.2025.102541) for specific experimental setups.

- Folder `AUFL` (models using Asymmetric Unified Focal Loss)

  Experiment 1: in folder `no_DA_subset1_cv`
  
  Experiment 3: in folder `DA_subset1_cv`

  Experiments 5 & 6: in folder `subsets1_2`

  Experiment 7: in folder `full_dataset`

- Folder `HL` (models using Hybrid Loss)

  Experiment 1: in folder `no_DA_subset1_cv`

  Experiment 3: in folder `DA_subset1_cv`

## 3D Models

Before using the 3D models for reproduction, please check the [codes](https://github.com/BYO-UPM/Craneal_CT/blob/main/Codes/3D_unets/3D_data_preprocess.ipynb) to stack the 2D CT images into 3D volume. All 3D models were obtained in Experiment 2 (see our [paper](https://doi.org/10.1016/j.compmedimag.2025.102541)) with cross-validation, using the Hybrid Loss. 

3D U-Net architectures with different encoders:

- Folder `vanilla`
- Folder `vgg16`
- Folder `resnet50`

