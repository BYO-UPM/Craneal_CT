from dataloaders.ct_test_dataloader import CATScansDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os


# Input path
#root = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/Dataset_PNGFiles/NoGroundTruth/Original_CT_images"
root = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/Dataset_PNGFiles/NoGroundTruth/Preprocessed_Same_Window"
folder_name = [f for f in sorted(os.listdir(root))]

# Ouput path
#output_root = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/Prediction_Results/Raw_PNG_CT/Default_windows"
output_root = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/Prediction_Results/Raw_PNG_CT/Same_window_preprocessed"
output_folder = [f for f in sorted(os.listdir(output_root))]

for idx_image in range(len(folder_name)):
    path = os.path.join(root, folder_name[idx_image])
    output_path = os.path.join(output_root, output_folder[idx_image])
    # Get the list of PNG files in the folder
    filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]

    # Define a transformation pipeline including the preprocessing function
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
        transforms.Normalize(mean=0, std=(1 / 255)),
    ])

    # Initialize CATScansDataset with the root directory and transformations
    test_dataset = CATScansDataset(root_dir=path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    ENCODER = 'vgg16'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=1, 
        activation=ACTIVATION,
        in_channels=1,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the best model
    modelname = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/DL_Models/Model_2D/Full_dataset/new_vgg2D_aug_win_fulldataset.pth"
    model.load_state_dict(torch.load(modelname))
    model.eval()

    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        in_image = inputs.detach().cpu().numpy()

        mask_prediction = model(inputs)
        mask_prediction = mask_prediction.detach().cpu().numpy()
        mask_prediction = mask_prediction > 0.5

        for j in range(mask_prediction.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(in_image[j,0,:,:], cmap="gray")
            ax[0].set_title("Mask Manual")
            ax[1].imshow(mask_prediction[j,0,:,:], cmap="gray")
            ax[1].set_title("Prediction Mask")
            output_name = f"Patient_{folder_name[idx_image]}_Slice_{i}_{j}.png"
            output_p = os.path.join(output_path, output_name)
            plt.savefig(output_p)
            plt.close()

# Check two images
#original_image, mask_image, patient_id, slice_number = full_dataset[0]
# Plot them
#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(original_image[0], cmap="gray")
#ax[0].set_title("Original Image")
#ax[1].imshow(mask_image[0], cmap="gray")
#ax[1].set_title("Mask Image")
#plt.show()
