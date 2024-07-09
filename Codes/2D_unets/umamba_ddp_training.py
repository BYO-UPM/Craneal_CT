'''
Still need modification, U-Mamba has a conflict with DDP.
'''

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from dataloaders.ct_da_dataloader import CATScansDataset, CustomAugmentation, AugmentedDataset
from losses.losses import FocalLossForProbabilities, AsymmetricUnifiedFocalLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.UMambaBot_2d import UMambaBot
from models.network_initialization import InitWeights_He

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
    ])
    full_dataset = CATScansDataset(root_dir="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/Internal Dataset", transform=transform)
    #custom_augmentation = CustomAugmentation()
    #train_dataset = AugmentedDataset(full_dataset, custom_augmentation)
    #train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    #train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, shuffle=False)
    train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler, shuffle=False)

    # Model setup
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
    model.apply(InitWeights_He(1e-2))
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
    focal_loss = FocalLossForProbabilities()
    aufl = AsymmetricUnifiedFocalLoss(from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    train_loss_list = []

    num_epochs = 5
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, masks, _, _ = data
            inputs, masks = inputs.to(rank), masks.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = aufl(outputs[:,:1,:,:], masks)
            #loss = dice_loss(outputs, masks)+focal_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)

        if rank == 0:
            print(f"Epoch {epoch + 1}, loss: {epoch_loss}")
            torch.save(model.module.state_dict(), f'/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/modeltest/mamba_{epoch+1}.pth')
        
        # Synchronize after each epoch
        dist.barrier()
    cleanup()

def main():
    world_size = 1  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
