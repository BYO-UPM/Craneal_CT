from Codes_pytorch.dataloaders.ct_window_dataloader import (
    CATScansDataset,
    CustomAugmentation,
    AugmentedDataset,
)
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from losses.losses import FocalLossForProbabilities
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
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

# Check two images
#original_image, mask_image, patient_id, slice_number = full_dataset[0]
# Plot them
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(original_image[0], cmap="gray")
#ax[0].set_title("Original Image")
#ax[1].imshow(mask_image[0], cmap="gray")
#ax[1].set_title("Mask Image")
#plt.show()

# Patient id list
patient_id = full_dataset.patient_id
unique_patient_id = list(set(patient_id))
print(f"Number of unique patients: {len(unique_patient_id)}")

# Set-up for cross-validation
cv_indx = 0

# Test set
test_patients = [unique_patient_id[cv_indx]]
    
# Validation set
validation_index = (cv_indx + 1) % len(unique_patient_id)
val_patients = [unique_patient_id[validation_index]]
    
# Training set
train_patients = [x for j, x in enumerate(unique_patient_id) if j != cv_indx and j != validation_index]

# Split the full dataset based on patient_idlen
train_dataset = [x for x in full_dataset if x[2] in train_patients]
val_dataset = [x for x in full_dataset if x[2] in val_patients]
test_dataset = [x for x in full_dataset if x[2] in test_patients]
    
# Instantiate the CustomAugmentation class
custom_augmentation = CustomAugmentation()
    
print("Applying the augmentation to the train dataset")
train_dataset = AugmentedDataset(train_dataset, custom_augmentation)
    
# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

