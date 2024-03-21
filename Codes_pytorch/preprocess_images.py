from dataloaders.ct_test_dataloader import (
    CATScansDataset,
    PreprocessWindow,
)
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
#from models.unet2d_vanilla import VanillaUNet2D
import numpy as np
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import csv
import os
#from losses.losses import AsymmetricUnifiedFocalLoss

# Path
path = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/quicktest"
filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
    transforms.Normalize(mean=0, std=(1 / 255)),
])

windowing = PreprocessWindow()

# Initialize CATScansDataset with the root directory and transformations
test_dataset = CATScansDataset(root_dir=path, transform=transform, window=windowing)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

fig, axs = plt.subplots(10, 4, figsize=(16, 40))  # 5 rows, 8 columns
fig.subplots_adjust(hspace=0.1, wspace=0.1)  # Adjust space between plots

for i, ax in enumerate(axs.flat):
    # Check if the list has enough images to fill all subplots
    if i < 40:
        original_image = test_dataset[i]
        ax.imshow(original_image[0], cmap='gray')  # Plot image, you can choose the colormap that suits your images
        ax.axis('off')  # Hide axes
    else:
        ax.axis('off')  # Hide axes for any subplot without an image

# Show the plot
plt.show()


