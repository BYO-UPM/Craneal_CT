# Model Comparison 

## Summary

Here we compare our model with various state-of-the-art medical image segmentation models, including UNETR [1] (2021), Swin UNETR [2] (2022), and nnU-Net v2 [3], U-Mamba [4], and U-KAN [5] (2024). Except for our model and U-KAN, all these models were implemented in the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework, initially released in 2018. We utilized nnU-Net framework for training and testing these models. Additionally, we extracted U-Mamba (Bot & Enc) from nnU-Net framework for further experiments, with the relevant code provided in this [link](https://github.com/BYO-UPM/Craneal_CT/tree/main/Codes/2D_unets/models). The experiments for U-KAN were conducted by modifying the original public [code](https://github.com/CUHK-AIM-Group/U-KAN).

None of these new models have been previously tested on datasets related to sinus CT, leaving their performance in segmenting osseous structures near paranasal sinuses unverified. This comparative experiment aims to validate our model's performance on the sinus CT dataset and evaluate the adaptability and accuracy of all these models in specific medical imaging scenarios, offering valuable insights for future research and practical applications.

Building on the original experiments described in our [paper](https://www.biorxiv.org/content/10.1101/2024.06.21.599833v1), we employed the DICE + Focal Loss (Hybrid Loss) and AUFL (Asymmetric Unified Focal loss) from our study, along with the DICE + BCE Loss used in the nnU-Net framework. These experiments allowed us to validate the models' performance and analyze the impact of different loss functions. Additionally, we also validated the impact of different data augmentation (DA) techniques with [our DA techniques](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/data_augmentation.md) and the [nnU-Net framework DA techniques](https://arxiv.org/abs/1904.08128) on model performance.

## Results

The comparison results are displayed in the [figure](https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/model_comparison.png) below, all results were obtained after completing post-processing. For detailed experimental settings, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2024.06.21.599833v1).

<div align="center">
    <img src="https://github.com/BYO-UPM/Craneal_CT/blob/main/Documents/Figures/model_comparison.png" width="100%" alt="results">
</div>

## Discussion

### Internal validation (with Subset 1)

- During the experiments using the nnU-Net framework on our internal dataset (Subset 1), we observed exceptional performance from U-Mamba, particularly U-Mamba Bot, achieving a high DICE score of 93.10±2.1. Results from the other three models were generally consistent.

- From the perspective of 2D and 3D structures, the 2D architecture demonstrated superior performance over the 3D architecture.

- Changes in the loss function did not significantly affect model performance under identical conditions.

- DA techniques enhanced model performance: applying DA techniques increased the DICE score by approximately 1.00.

- Our models converged within 15 epochs, contrasting with the average 80 epochs required by the other six models for training completion. This discrepancy not only prolonged training time but also necessitated more computational resources and robust GPU support. -----We are addressing this by increasing the learning rate.-----

- The Mamba model consumes substantial memory, particularly for U-Mamba Enc, thereby increasing computational costs.

- ......

### External validation (with complete Subsets 1, 2 and 3)

- ......

## References

[1] A. Hatamizadeh, Y. Tang, et al. UNETR: Transformers for 3D Medical Image Segmentation. arXiv:2103.10504. (2021).

[2] A. Hatamizadeh, V. Nath	Swin, et al. UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv:2201.01266. (2022).

[3] Isensee F., Wald T., et al. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv preprint arXiv:2404.09556. (2024).

[4] J. Ma, F. Li, et al. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation. arXiv:2401.04722. (2024).

[5] C. Li, X. Liu, et al. U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation. arXiv:2406.02918. (2024).
