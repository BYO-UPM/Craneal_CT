from dataloaders.ct_randomcrop_dataloader import (
#from dataloaders.ct_blur_dataloader import (
    CATScansDataset,
    CustomAugmentation,
    AugmentedDataset,
)
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from losses.losses import FocalLossForProbabilities
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import csv
#from losses.losses import AsymmetricUnifiedFocalLoss

# Path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed"

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
    transforms.Normalize(mean=0, std=(1 / 255)),
])

# Initialize CATScansDataset with the root directory and transformations
full_dataset = CATScansDataset(root_dir=path, transform=transform)

# Create the train dataloaders
custom_augmentation = CustomAugmentation()
train_dataset = AugmentedDataset(full_dataset, custom_augmentation)
print("Applying the augmentation to the train dataset")

original_image, mask_image, patient_id, slice_number = train_dataset[3]
# Plot them
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_image[0], cmap="gray")
ax[0].set_title("Original Image")
ax[1].imshow(mask_image[0], cmap="gray")
ax[1].set_title("Mask Image")
plt.show()