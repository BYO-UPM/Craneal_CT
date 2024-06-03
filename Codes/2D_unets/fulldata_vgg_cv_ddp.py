import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from segmentation_models_pytorch import Unet
import segmentation_models_pytorch as smp
from dataloaders.ct_randomcrop_dataloader import CATScansDataset, CustomAugmentation, AugmentedDataset
from losses.losses import FocalLossForProbabilities
from losses.losses import AsymmetricUnifiedFocalLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import csv

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Transform and dataset setup
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(mean=0, std=(1 / 255)),
    ])
    full_dataset = CATScansDataset(root_dir="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed", transform=transform)
    
    patient_id = full_dataset.patient_id
    unique_patient_id = list(set(patient_id))
    unique_patient_id.sort()
    #print(f"Number of unique patients: {len(unique_patient_id)}")

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
        
        # Distributed data loading
        train_dataset = AugmentedDataset(train_dataset, custom_augmentation)
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, shuffle=False)
        
        # Create the dataloaders
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Model setup
        model = Unet(encoder_name='vgg16', encoder_weights='imagenet', classes=1, activation='sigmoid', in_channels=1)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])

        # Loss and optimizer
        dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
        focal_loss = FocalLossForProbabilities()
        aufl = AsymmetricUnifiedFocalLoss(from_logits=True)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Training loop
        train_loss_list = []

        num_epochs = 10
        train_loss_list = []
        val_loss_list = []
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, masks, _, _ = data
                inputs, masks = inputs.to(rank), masks.to(rank)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = aufl(outputs, masks)
                #loss = dice_loss(outputs, masks)+focal_loss(outputs, masks)
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
                inputs, masks = inputs.to(rank), masks.to(rank)

                # Forward
                outputs = model(inputs)
                loss = aufl(outputs, masks)
                #loss = dice_loss(outputs, masks)+focal_loss(outputs, masks)
                
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
                    modelname = f"vgg2D_unified_randomcrop_cv_{cv_indx}.pth"
                    torch.save(model.state_dict(), modelname)
            
            # Synchronize after each epoch
            dist.barrier()
        
        # Save information for training and validation losses
        # New csv file
        filename = f"vgg2D_unified_randomcrop_cv_{cv_indx}.csv"
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
            inputs, masks = inputs.to(rank), masks.to(rank)

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

    cleanup()

def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
