#from dataloaders.ct_aug_dataloader import (
from dataloaders.ct_window_dataloader import (
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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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

dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
focal_loss = FocalLossForProbabilities()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_list = []
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
        # loss = AsymmetricUnifiedFocalLoss(from_logits=True)(mask_prediction, masks)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")
    train_loss_list.append(running_loss / len(train_loader))

    # Save the best model
    if epoch == 0:
        best_loss = running_loss / len(train_loader)
    else:
        if running_loss / len(train_loader) < best_loss:
            best_loss = running_loss / len(train_loader)
            print(f"Best model so far, saving the model at epoch {epoch + 1}")
            modelname = f"vgg2D_aug_win_fulldataset.pth"
            torch.save(model.state_dict(), modelname)

# Save information for training and validation losses
# New csv file
filename = f"loss_vgg2D_aug_win_fulldataset.csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Train Loss'])
    for train_val in zip(train_loss_list):
        writer.writerow(list(train_val[0:1]))

# Check two images
#original_image, mask_image, patient_id, slice_number = full_dataset[0]
# Plot them
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(original_image[0], cmap="gray")
#ax[0].set_title("Original Image")
#ax[1].imshow(mask_image[0], cmap="gray")
#ax[1].set_title("Mask Image")
#plt.show()
