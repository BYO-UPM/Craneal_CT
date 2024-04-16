import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from segmentation_models_pytorch import Unet
from dataloaders.ct_zoomin_dataloader import CATScansDataset, CustomAugmentation, AugmentedDataset
from losses.losses import FocalLossForProbabilities
import matplotlib.pyplot as plt


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
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=(1 / 255)),
    ])
    full_dataset = CATScansDataset(root_dir="../CAT_scans_Preprocessed", transform=transform)
    custom_augmentation = CustomAugmentation()
    train_dataset = AugmentedDataset(full_dataset, custom_augmentation)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, shuffle=False)

    # Model setup
    model = Unet(encoder_name='vgg16', encoder_weights='imagenet', classes=1, activation='sigmoid', in_channels=1)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    dice_loss = FocalLossForProbabilities()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    train_loss_list = []

    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, masks, _, _ = data
            inputs, masks = inputs.to(rank), masks.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)

        if rank == 0:
            print(f"Epoch {epoch + 1}, loss: {epoch_loss}")
            torch.save(model.module.state_dict(), f'model_epoch_{epoch+1}.pth')

    if rank == 0:
        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_loss_list, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()

    cleanup()
    
    cleanup()

def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
