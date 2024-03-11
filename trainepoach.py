from dataloaders.pytorch_ct_dataloader import (
    CATScansDataset,
    CustomAugmentation,
    AugmentedDataset,
)
from losses.losses import FocalLossForProbabilities
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm
import segmentation_models_pytorch.utils
import csv
#from losses.losses import AsymmetricUnifiedFocalLoss


# Path
path = "CAT_scans_Preprocessed"

# Common transformation, normalize between 0 and 1
preprocess_input = get_preprocessing_fn('vgg16', pretrained='imagenet')  

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
cv_DICE = []

def modified_dataloader(original_dataloader):
    for data in original_dataloader:
        yield data[:2]  

# 7 patients to train, 1 to val and 1 to test
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

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    # Training loop

    # Loss and optimizer
    # Dice loss and focal loss
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode="binary")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(modified_dataloader(train_loader))
        valid_logs = valid_epoch.run(modified_dataloader(val_loader))
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        
        # Save information for training and validation losses
        # New csv file
        filename = f"loss_vanilla2D_aug_cv_{cv_indx}.csv"
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Train Loss'] + [''] * 10 + ['Validation Loss'])
            for train_val in zip(train_loss_list, val_loss_list):
                writer.writerow(list(train_val[0:1]) + [''] * 10 + list(train_val[1:]))


    # Evaluate the model in test with DICE score
 
    # Load the best model
    model.load_state_dict(torch.load(modelname))
    model.eval()

    # DICE score
    dice_score = []

    for i, data in enumerate(test_loader):
        inputs, masks, _, _ = data
        inputs, masks = inputs.to(device), masks.to(device)

        # Forward
        mask_prediction = model(inputs)
        mask_prediction = torch.sigmoid(mask_prediction)
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
