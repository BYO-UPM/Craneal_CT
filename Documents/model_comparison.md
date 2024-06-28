# Model Comparison 

We used the **unaugmented** Subset 1 for this comparison test. Our model architecture was: U-Net with encoder VGG16 using asymmetric unified focal loss.

Our model is compared with nnUNet version 2 (nnUNetv2) [1], UNETR [2], SwinUNETR [3], U-Mamba [4] and U-KAN [5]. The results are shown in the table below:

|           | Our model   |  nnUNetV2  |   UNETR    |  SwinUNETR  |   U-Mamba_Bot   |   U-Mamba_Enc   |   U-KAN   |
|:---------:|:-----------:|:----------:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| DICE Score| 91.73±1.7   | 91.99±2.0  | 91.75±1.7  |  91.85±2.2    |  93.10±2.1  |  92.82±2.3   |  91.58±4.3   |

## References

[1] Isensee F., Wald T., et al. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv preprint arXiv:2404.09556. (2024).

[2] M. Jorge Cardoso, Wenqi Li, et al. MONAI: An open-source framework for deep learning in healthcare. arXiv:2211.02701v1. (2022).

[3] A. Hatamizadeh, V. Nath	Swin, et al. UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv:2201.01266. (2022).

[4] J. Ma, F. Li, et al. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation. arXiv:2401.04722. (2024).

[5] C. Li, X. Liu, et al. U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation. arXiv:2406.02918. (2024).
