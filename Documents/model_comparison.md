# Model Comparison 

We used the **unaugmented** Subset 1 for this comparison test. Our model architecture was: U-Net with encoder VGG16 using asymmetric unified focal loss.

Our model is compared with nnUNet version 2 (nnUNetv2), UNETR and U-Mamba. The results are shown in the table below:

|           | Our model   |  nnUNetV2  |   UNETR    |   U-Mamba   |
|:---------:|:-----------:|:----------:|:----------:|:-----------:|
| DICE Score| 91.73±1.7   | 91.99±2.0  |    ±       |     ±       |
