# Available Models

Due to the file size limitation in GitHub, please view our available models at this [link](https://drive.google.com/drive/folders/19Clgax8sH59DEnHt1XK6PBL_eQEiHX_S?usp=sharing).

## Final Model

The final model is stored in folder `Final_Model`.

The final model was obtained by training the true labels of Subsets 1 & 2, and all the pseudo-labels of Subsets 2 & 3. It can be used for further **transfer learning** applications. DO NOT use this model for testing on our datasets, as the training set already contained all CT slices in our datasets.

## 2D Models

You can use these models for **reproduction**. See our [paper](https://github.com/BYO-UPM/Craneal_CT) for specific experimental setups.

- Folder `AUFL` (models using Asymmetric Unified Focal Loss)

  Experiment 1: in folder `no_DA_subset1_cv`
  
  Experiment 3: in folder `DA_subset1_cv`

  Experiments 5 & 6: in folder `subsets1_2`

  Experiment 7: in folder `full_dataset`

- Folder `HL` (models using Hybrid Loss)

  Experiment 1: in folder `no_DA_subset1_cv`

  Experiment 3: in folder `DA_subset1_cv`

## 3D Models

The corresponding 3D models can be obtained by running these [codes](https://github.com/BYO-UPM/Craneal_CT/tree/main/Codes/3D_unets) for training, and due to the large total size of the 3D models (more than 15 GB), they are not available for direct download.


