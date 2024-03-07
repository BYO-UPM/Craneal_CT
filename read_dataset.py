from dataloaders.pytorch_ct_dataloader import CATScansDataset, CustomAugmentation
from matplotlib import pyplot as plt
from torchvision import transforms
from models.unet import VanillaUNet2D

import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


# Path
path = "CAT_scans_Preprocessed"

# # Instantiate the CustomAugmentation class
# custom_augmentation = CustomAugmentation()

# Common transformation, normalize between 0 and 1
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=(1 / 255)),
    ]
)

# Initialize CATScansDataset with the root directory and transformations
full_dataset = CATScansDataset(root_dir=path, transform=transform)

# Check two images
original_image, mask_image, patient_id, slice_number = full_dataset[0]

# Plot them
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_image[0], cmap="gray")
ax[0].set_title("Original Image")
ax[1].imshow(mask_image[0], cmap="gray")
ax[1].set_title("Mask Image")
plt.show()


# Patient id list
patient_id = full_dataset.patient_id

unique_patient_id = list(set(patient_id))
print(f"Number of unique patients: {len(unique_patient_id)}")

# Randomly, 7 patients to train, 1 to val and 1 to test
import random

random.seed(42)
random.shuffle(unique_patient_id)

train_patients = unique_patient_id[:7]
val_patients = unique_patient_id[7:8]
test_patients = unique_patient_id[8:]

# Generate dataloaders
from torch.utils.data import DataLoader

# Split the full dataset based on patient_id
train_dataset = [x for x in full_dataset if x[2] in train_patients]
val_dataset = [x for x in full_dataset if x[2] in val_patients]
test_dataset = [x for x in full_dataset if x[2] in test_patients]

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Instantiate the model
model = VanillaUNet2D(1, 512, 512)

# Training loop
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
from losses.losses import AsymmetricUnifiedFocalLoss

# Loss and optimizer
# Dice loss and focal loss
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
focal_loss = smp.losses.FocalLoss(mode="binary")
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


val_loss_list = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, masks, _, _ = data
        inputs, masks = inputs.to(device), masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        mask_prediction = model(inputs)
        diceloss = dice_loss(mask_prediction, masks)
        focalloss = focal_loss(mask_prediction, masks)
        loss = diceloss + focalloss
        loss = AsymmetricUnifiedFocalLoss(from_logits=True)(mask_prediction, masks)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")

    # validation loop
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(val_loader):
        inputs, masks, _, _ = data
        inputs, masks = inputs.to(device), masks.to(device)

        # Forward
        mask_prediction = model(inputs)
        diceloss = dice_loss(mask_prediction, masks)
        focalloss = focal_loss(mask_prediction, masks)
        loss = diceloss + focalloss
        # loss = AsymmetricUnifiedFocalLoss(from_logits=True)(mask_prediction, masks)

        # Print statistics
        running_loss += loss.item()

    print(f"Validation loss: {running_loss / len(val_loader)}")
    val_loss_list.append(running_loss / len(val_loader))

    # Save the best model
    if epoch == 0:
        best_loss = running_loss / len(val_loader)
    else:
        if running_loss / len(val_loader) < best_loss:
            best_loss = running_loss / len(val_loader)
            print(f"Best model so far, saving the model at epoch {epoch + 1}")
            torch.save(model.state_dict(), "best_model.pth")


# Evaluate the model in test with DICE score
import numpy as np

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))
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

print(f"Mean DICE score: {np.mean(dice_score)}")
print(f"Std DICE score: {np.std(dice_score)}")
