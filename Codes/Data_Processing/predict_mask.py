from ct_test_dataloader import CATScansDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
import os
import re

def extract_last_number(filename):
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

# Input path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Image"
filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]
filenames = sorted(filenames, key=extract_last_number)

# Ouput path
output_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/fold"

# Define a transformation pipeline including the preprocessing function
transform = transforms.Compose([
    transforms.ToTensor(),
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
modelname = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Codes/model.pth"
state_dict = torch.load(modelname)

# delete "module." 
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # delete
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# load the model
model.load_state_dict(new_state_dict)
model.eval()
inx_s = 1

for i, data in enumerate(test_loader):
    inputs = data
    inputs = inputs.to(device)
    in_image = inputs.detach().cpu().numpy()

    mask_prediction = model(inputs)
    mask_prediction = mask_prediction.detach().cpu().numpy()
    mask_prediction = mask_prediction > 0.5

    for j in range(mask_prediction.shape[0]):
        output_name = f"P01_predict_{inx_s}.png"
        inx_s = inx_s+1
        output_p = os.path.join(output_path, output_name)
        plt.imsave(output_p, mask_prediction[j,0,:,:], cmap='gray', format='png')
        plt.close() 
