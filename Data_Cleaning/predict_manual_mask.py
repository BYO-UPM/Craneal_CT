from ct_test_dataloader import CATScansDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os
import re

def extract_last_number(filename):
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

# Input path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Original"
filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]
filenames = sorted(filenames, key=extract_last_number)

# Ouput path
output_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/dddd"

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
modelname = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/vgg2D_crop_fulldataset.pth"
model.load_state_dict(torch.load(modelname))
model.eval()
inx_s = 0

for i, data in enumerate(test_loader):
    inputs = data
    inputs = inputs.to(device)
    in_image = inputs.detach().cpu().numpy()

    mask_prediction = model(inputs)
    mask_prediction = mask_prediction.detach().cpu().numpy()
    mask_prediction = mask_prediction > 0.5

    for j in range(mask_prediction.shape[0]):
        output_name = f"P16_new_auto_mask_{inx_s}.png"
        inx_s = inx_s+1
        output_p = os.path.join(output_path, output_name)
        plt.imsave(output_p, mask_prediction[j,0,:,:], cmap='gray', format='png')
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
