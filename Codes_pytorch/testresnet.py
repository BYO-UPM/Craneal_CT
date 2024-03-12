from dataloaders.pytorch_ct_dataloader import (
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
path = "CAT_scans_Preprocessed"

# Custom transform to convert a grayscale image to RGB
class GrayscaleToRGBTransform:
    def __call__(self, x):
        # x is a grayscale image with shape [1, H, W]
        # We repeat the grayscale channel 3 times to make it RGB
        return x.repeat(3, 1, 1)

# Common transformation, normalize between 0 and 1
preprocess_input = get_preprocessing_fn('vgg16', pretrained='imagenet')  

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    #transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to tensor
    #transforms.Lambda(lambda x: x.mul(255).byte()),  # Scale to [0, 255] and convert to uint8
    #GrayscaleToRGBTransform(),
    transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
    transforms.Normalize(mean=0, std=(1 / 255)),
    #transforms.Lambda(lambda x: preprocess_input(x.transpose(1, 2, 0).numpy())),  # Apply preprocessing
    #transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1).float())),  # Back to tensor
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
cv_DICE = []

for cv_indx in range(len(unique_patient_id)):
    #random.seed(42)
    #random.shuffle(unique_patient_id)
    #train_patients = unique_patient_id[:7]
    #val_patients = unique_patient_id[7:8]
    #test_patients = unique_patient_id[8:]

    # Test set
    test_patients = [unique_patient_id[cv_indx]]
    
    # Validation set
    validation_index = (cv_indx + 1) % len(unique_patient_id)
    val_patients = [unique_patient_id[validation_index]]
    
    # Training set
    train_patients = [x for j, x in enumerate(unique_patient_id) if j != cv_indx and j != validation_index]

    # Split the full dataset based on patient_id
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

    ENCODER = 'resnet50'
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Evaluate the model in test with DICE score
 
    # Load the best model
    modelname = f"resnet2D_aug_cv_{cv_indx}.pth"
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
    
    cv_DICE.append(np.mean(dice_score))
    print(f"Mean DICE score: {np.mean(dice_score)}")
    print(f"Std DICE score: {np.std(dice_score)}")

print(f"Cross-validation END: Mean DICE is: {np.mean(cv_DICE)}")
print(f"                      std DICE is: {np.std(cv_DICE)}")