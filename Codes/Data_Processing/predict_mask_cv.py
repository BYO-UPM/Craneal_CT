from ct_test_dataloader import CATScansDataset
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
#from unet2d_vanilla import VanillaUNet2D
import torch
import segmentation_models_pytorch as smp
import os
import re

def extract_last_number(filename):
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

# Input path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed"
#path = sorted(os.listdir(path))
path = ['S090899',
 'S090934',
 'S090946',
 'S090947',
 'S090957',
 'S090967',
 'S090977',
 'S090979',
 'S090981']

for i in range(len(path)):
    input = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed/{path[i]}/Original"
    
    # Ouput path
    output_path = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/result_e3_AUFL_DA_cv/cv{i+1}"

    # Define a transformation pipeline including the preprocessing function
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Initialize CATScansDataset with the root directory and transformations
    test_dataset = CATScansDataset(root_dir=input, transform=transform)
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

    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    # Load the best model
    
    modelname = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/e123_models2D/AUFL/DA_cv/vgg2D_unified_randomcrop_cv_{i}.pth"
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

    for y, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        in_image = inputs.detach().cpu().numpy()

        mask_prediction = model(inputs)
        mask_prediction = mask_prediction.detach().cpu().numpy()
        mask_prediction = mask_prediction > 0.5

        for j in range(mask_prediction.shape[0]):
            output_name = f"P0{i+1}_vgg_cv{i+1}_{inx_s}.png"
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
