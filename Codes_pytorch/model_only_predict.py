from dataloaders.ct_aug_dataloader import (
    CATScansDataset,
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
unique_patient_id.sort()
print(f"Number of unique patients: {len(unique_patient_id)}")

# Set-up for cross-validation
cv_DICE = []

for cv_indx in range(len(unique_patient_id)):
    # Test set
    test_patients = [unique_patient_id[cv_indx]]

    # Split the full dataset based on patient_id
    test_dataset = [x for x in full_dataset if x[2] in test_patients]
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    ENCODER = 'vgg16'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'

    # Instantiate the model
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=1, 
        activation=ACTIVATION,
        in_channels=1,
    )

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model in test with DICE score
 
    # Load the best model
    modelname = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Codes_pytorch/vgg2D_aug_unified_cv_{cv_indx}.pth"
    model.load_state_dict(torch.load(modelname))
    model.eval()

    # DICE score
    dice_score = []

    for i, data in enumerate(test_loader):
        inputs, masks, _, _ = data
        inputs, masks = inputs.to(device), masks.to(device)

        # Forward
        mask_prediction = model(inputs)
        #mask_prediction = torch.sigmoid(mask_prediction)
        mask_prediction = mask_prediction.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        # DICE score
        mask_prediction = mask_prediction > 0.5

        intersection = np.sum(mask_prediction * masks)
        union = np.sum(mask_prediction) + np.sum(masks)
        dice = (2 * intersection) / (union + 1e-8)
        dice_score.append(dice)

        '''for j in range(16):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(masks[j,0,:,:], cmap="gray")
            ax[0].set_title("Mask Manual")
            ax[1].imshow(mask_prediction[j,0,:,:], cmap="gray")
            ax[1].set_title("Prediction Mask")
            resultpath = f"/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/Prediction_Results/vgg2D_aug_{i}_{j}.png"
            plt.savefig(resultpath)
            plt.close()'''
    
    cv_DICE.append(np.mean(dice_score))
    print(f"Mean DICE score: {np.mean(dice_score)}")
    print(f"Std DICE score: {np.std(dice_score)}")

print(f"Cross-validation END: Mean DICE is: {np.mean(cv_DICE)}")
print(f"                      std DICE is: {np.std(cv_DICE)}")
