from dataloaders.ct_da_dataloader import (
    CATScansDataset,
    CustomAugmentation,
    AugmentedDataset,
)
import sys
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
from losses.losses import AsymmetricUnifiedFocalLoss
from models.UMambaBot_2d import UMambaBot
from models.UMambaEnc_2d import UMambaEnc
from models.network_initialization import InitWeights_He
import argparse


parser = argparse.ArgumentParser(description='Train and evaluate a model for CT scans')
parser.add_argument('--log_file', type=str, required=True, help='Path to the log file', default='./modeltest/enc_mamba_P38.txt')
parser.add_argument('--model_type', type=str, required=True, choices=['MambaEnc', 'MambaBot'], help='Model type to use')
parser.add_argument('--gpu', type=int, required=True, help='GPU index to use')

args = parser.parse_args()

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(args.log_file)
sys.stderr = Logger(args.log_file)

# Set GPU device
torch.cuda.set_device(args.gpu)



# Path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2"

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize CATScansDataset with the root directory and transformations
full_dataset = CATScansDataset(root_dir=path, transform=transform)


# Patient id list
patient_id = full_dataset.patient_id
unique_patient_id = list(set(patient_id))
unique_patient_id.sort()
print(f"Number of unique patients: {len(unique_patient_id)}")

# Set-up for cross-validation
cv_DICE = []

# 7 patients to train, 1 to val and 1 to test
for cv_indx in range(len(unique_patient_id)):
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
    train_dataset = AugmentedDataset(train_dataset, custom_augmentation)
    
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # U-Mamba Bot
    if args.model_type == 'MambaBot':
        model = UMambaBot(
            input_channels = 1,
            n_stages = 7,
            features_per_stage = [32, 64, 128, 256, 512, 512, 512],
            conv_op = nn.Conv2d,
            kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2],
            num_classes = 2,
            n_conv_per_stage_decoder = [2, 2, 2, 2, 2, 2],
            conv_bias = True,
            norm_op = nn.InstanceNorm2d,
            norm_op_kwargs = {'eps': 1e-5, 'affine': True},
            dropout_op = None,
            dropout_op_kwargs = None,
            nonlin = nn.LeakyReLU,
            nonlin_kwargs = {'inplace': True},
            deep_supervision = False,
        )
    else:
        # U-Mamba Enc
        model = UMambaEnc(
            input_size = [512, 512],
            input_channels = 1,
            n_stages = 7,
            features_per_stage = [32, 64, 128, 256, 512, 512, 512],
            conv_op = nn.Conv2d,
            kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2],
            num_classes = 2,
            n_conv_per_stage_decoder = [2, 2, 2, 2, 2, 2],
            conv_bias = True,
            norm_op = nn.InstanceNorm2d,
            norm_op_kwargs = {'eps': 1e-5, 'affine': True},
            dropout_op = None,
            dropout_op_kwargs = None,
            nonlin = nn.LeakyReLU,
            nonlin_kwargs = {'inplace': True},
            deep_supervision = False
        )

    # Model init
    model.apply(InitWeights_He(1e-2))

    # Optimizer and loss function
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
    focal_loss = FocalLossForProbabilities()
    aufl = AsymmetricUnifiedFocalLoss(from_logits=False)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training loop
    num_epochs = 20
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_loss_list = []
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
            #diceloss = dice_loss(mask_prediction, masks)
            #focalloss = focal_loss(mask_prediction, masks)
            #loss = diceloss + focalloss
            loss = aufl(mask_prediction[:,:1,:,:], masks)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")
        train_loss_list.append(running_loss / len(train_loader))

        # validation loop
        model.eval()
        running_loss = 0.0
        for i, data in enumerate(val_loader):
            inputs, masks, _, _ = data
            inputs, masks = inputs.to(device), masks.to(device)

            # Forward
            mask_prediction = model(inputs)
            #diceloss = dice_loss(mask_prediction, masks)
            #focalloss = focal_loss(mask_prediction, masks)
            #loss = diceloss + focalloss
            loss = aufl(mask_prediction[:,:1,:,:], masks)

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
                modelname = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_e5_cv_{cv_indx+1}.pth"
                torch.save(model.state_dict(), modelname)
    
    # Save information for training and validation losses
    # New csv file
    filename = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_e5_cv_{cv_indx}.csv"
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
        #mask_prediction = torch.sigmoid(mask_prediction)
        mask_prediction = mask_prediction[:,:1,:,:].detach().cpu().numpy()
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
