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
import os


parser = argparse.ArgumentParser(description='Train and evaluate a model for CT scans')
parser.add_argument('--model_type', type=str,  choices=['MambaEnc', 'MambaBot'], help='Model type to use', default="MambaEnc")
parser.add_argument('--gpu', type=int,  help='GPU index to use', choices=[0, 1, 2, 3], default=1)
parser.add_argument('--test_id', type=str,  help='Patient id for testing', choices=["P38", "P16", "P30", "P13"], default="P38")
parser.add_argument('--batch_size', type=int,  help='Batch size', default=16)
parser.add_argument('--train', type=bool, help="If true, training the model. If false, just inference", default=False)

# Example on how to run this
# python umamba_training_exp5-7.py --model_type MambaEnc --gpu 1 --test_id P38

args = parser.parse_args()

# print args
print(args)

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

log_file = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/{args.model_type}_e5_model_to_test_{args.test_id}.log"
sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

# Set GPU device
torch.cuda.set_device(args.gpu)



# Path
path_internal =  "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/Internal Dataset"
path_external = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/External Dataset"

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize CATScansDataset with the root directory and transformations
internal_dataset = CATScansDataset(root_dir=path_internal, transform=transform)
external_dataset_one = CATScansDataset(root_dir=path_external, transform=transform)
external_dataset_two = CATScansDataset(root_dir=path_external, transform=transform)

# USe remove_patients to remove the test_id rfrom external_dataset
external_dataset_to_train = external_dataset_one.remove_patients([args.test_id]) 

# Add to the internal dataset the external dataset
train_dataset = internal_dataset.add_patient(set(external_dataset_to_train.patient_id), root_dir=path_external)

# Test dataset: remove all ids but the test_id
test_dataset = external_dataset_two.remove_patients(set(external_dataset_to_train.patient_id))

# Check that patient_id is not overlapping
train_patient_id = train_dataset.patient_id
test_patient_id = test_dataset.patient_id
assert len(set(train_patient_id).intersection(set(test_patient_id))) == 0


# Patient id list
patient_id = train_dataset.patient_id
unique_patient_id = list(set(patient_id))
unique_patient_id.sort()
print(f"Number of unique patients: {len(unique_patient_id)}")

# Instantiate the CustomAugmentation class
custom_augmentation = CustomAugmentation()
train_dataset = AugmentedDataset(train_dataset, custom_augmentation)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


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



# Optimizer and loss function
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
focal_loss = FocalLossForProbabilities()
aufl = AsymmetricUnifiedFocalLoss(from_logits=False)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

model.to(device)

if args.train:
    # Model init
    model.apply(InitWeights_He(1e-2))

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader)):
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

        # Save the best model
        print(f"Best model so far, saving the model at epoch {epoch + 1}")
        modelname = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_e5_model_to_test_{args.test_id}.pth"
        torch.save(model.state_dict(), modelname)

    # Save information for training and validation losses
    # New csv file
    filename = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_e5_model_to_test_{args.test_id}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Store train loss
        writer.writerow(train_loss_list)


# Now test the model
model.eval()
test_loss = 0.0
for i, data in enumerate(test_loader):
    inputs, masks, _, _ = data
    inputs, masks = inputs.to(device), masks.to(device)

    mask_prediction = model(inputs)
    loss = aufl(mask_prediction[:,:1,:,:], masks)
    test_loss += loss.item()

    # Save in "modeltest/mamba_e5_cv_{test_id}/" folder the images predicted
    mask_prediction = mask_prediction.detach().cpu().numpy()
    mask_prediction = np.argmax(mask_prediction, axis=1)
    mask_prediction = mask_prediction.squeeze()
    mask_prediction = mask_prediction.astype(np.uint8)
    mask_prediction = mask_prediction * 255

    # Make the directory to save the predictions using model name and patient in test
    save_dir = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_e5_cv_{args.test_id}/"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.imsave(os.path.join(save_dir, f"prediction_{i}.png"), mask_prediction, cmap='gray')

print(f"Test loss: {test_loss / len(test_loader)}")



