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
path = f"/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/quicktest"
filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
    transforms.Normalize(mean=0, std=(1 / 255)),
])

windowing = PreprocessWindow()

# Initialize CATScansDataset with the root directory and transformations
test_dataset = CATScansDataset(root_dir=path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

# Check two images
#original_image, mask_image, patient_id, slice_number = full_dataset[0]
# Plot them
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(original_image[0], cmap="gray")
#ax[0].set_title("Original Image")
#ax[1].imshow(mask_image[0], cmap="gray")
#ax[1].set_title("Mask Image")
#plt.show()
for cv_indx in range(1):

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

    #model = VanillaUNet2D(1, 512, 512)
    device = torch.device("cpu")
    model.to(device)
 
    # Load the best model
    modelname = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/resnet2D_window_cv_{cv_indx}.pth"
    #modelname = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/vanilla2D_window_cv_0.pth"
    model.load_state_dict(torch.load(modelname))
    model.eval()

    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)

        # Forward
        mask_prediction = model(inputs)
        mask_prediction = mask_prediction.detach().cpu().numpy()
        mask_prediction = mask_prediction > 0.5


