import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from segmentation_models_pytorch import Unet
from dataloaders.ct_da_dataloader import CATScansDataset, CustomAugmentation, AugmentedDataset, RandomScansDataset
from losses.losses import AsymmetricUnifiedFocalLoss
from tqdm import tqdm
import random

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
    full_dataset = CATScansDataset(root_dir="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed", transform=transform)
    random_data_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset3"
    custom_augmentation = CustomAugmentation()
    train_dataset = AugmentedDataset(full_dataset, custom_augmentation)
    
    # Model setup
    model = Unet(encoder_name='vgg16', encoder_weights='imagenet', classes=1, activation='sigmoid', in_channels=1)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # load the model
    path_model = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/e567_AUFLmodels/fold3_30/semi_fullset/e7_fold3_1.pth"
    state_dict = torch.load(path_model)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = "module." + key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    # Loss and optimizer
    #dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
    #focal_loss = FocalLossForProbabilities()
    aufl = AsymmetricUnifiedFocalLoss(from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    train_loss_list = []

    num_epochs = 2
    for epoch in tqdm(range(num_epochs)):

        random_subset3 = random.sample(os.listdir(random_data_path), 12)
        random_dataset = RandomScansDataset(root_dir=random_data_path, patient_idx=random_subset3, transform=transform)
        random_DA_dataset = AugmentedDataset(random_dataset, custom_augmentation)

        combined_dataset = ConcatDataset([train_dataset, random_DA_dataset])
        train_sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(combined_dataset, batch_size=64, sampler=train_sampler, shuffle=False)
        
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, masks, _, _ = data
            inputs, masks = inputs.to(rank), masks.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = aufl(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)

        if rank == 0:
            print(f"Epoch {epoch + 1}, loss: {epoch_loss}")
            torch.save(model.module.state_dict(), f'/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/e567_AUFLmodels/fold3_30/semi_fullset/e7_fold3_{epoch+2}.pth')
        
        # Synchronize after each epoch
        dist.barrier()

    #if rank == 0:
        # Plot the training losses
    #    plt.figure(figsize=(10, 5))
    #    plt.plot(range(1, num_epochs+1), train_loss_list, label='Training Loss')
    #    plt.xlabel('Epochs')
    #    plt.ylabel('Loss')
    #    plt.title('Training Loss Over Epochs')
    #    plt.legend()
    #    plt.grid(True)
    #    plt.savefig('model4_par_training_loss.png')
    #    plt.close()
    cleanup()

def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
