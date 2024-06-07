from skimage.io import imread
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import re
from torchvision import transforms
from ct_test_dataloader import CATScansDataset
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from scipy.ndimage import label, generate_binary_structure
import cc3d
from scipy.ndimage import binary_fill_holes


def remove_floating_objects(volume):
    # Generate a binary structure for 3D connectivity (26-connected)
    struct = generate_binary_structure(3, 3)

    # Label connected components
    labeled_volume, num_features = label(volume, structure=struct)

    # Find the largest connected component
    component_sizes = np.bincount(labeled_volume.ravel())
    largest_component_label = component_sizes[1:].argmax() + 1

    # Create a mask for the largest connected component
    largest_component = labeled_volume == largest_component_label

    return largest_component


# DICE
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice


# Extract last number from a string
def extract_last_number(filename):
    matches = re.findall(r"\d+", filename)
    return int(matches[-1]) if matches else 0


# Paths
patient_id = "P13"
path_original = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/e5_train_30slices/{patient_id}/Original"
original_img = sorted(os.listdir(path_original), key=extract_last_number)
if '.DS_Store' in original_img:
        original_img.remove('.DS_Store')
path_manual = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/e5_train_30slices/{patient_id}/Mask"
manual_img = sorted(os.listdir(path_manual), key=extract_last_number)
manual_img = [file for file in manual_img if file.endswith('.png')]
#path_model = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/e567_AUFLmodels/fold3_30/sup/exp_5_fold3_4.pth"
path_model = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/e567_AUFLmodels/fold2_13/semi/e6_fold2_7.pth"

#path_output = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/test/{patient_id}/result_e5_sup"
#path_output_post = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/test/{patient_id}/result_e5_sup_post"
path_output = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/test/{patient_id}/result_e6_semi"
path_output_post = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/subset2/test/{patient_id}/result_e6_semi_post"

transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize CATScansDataset with the root directory and transformations
test_dataset = CATScansDataset(root_dir=path_original, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = smp.Unet(
    encoder_name='vgg16', 
    encoder_weights='imagenet', 
    classes=1, 
    activation='sigmoid',
    in_channels=1,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# load the model
state_dict = torch.load(path_model)

# delete "module." 
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # delete
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)
model.eval()

for i, data in enumerate(test_loader):
    inputs = data
    inputs = inputs.to(device)
    in_image = inputs.detach().cpu().numpy()

    mask_prediction = model(inputs)
    mask_prediction = mask_prediction.detach().cpu().numpy()
    mask_prediction = mask_prediction > 0.5

    for j in range(mask_prediction.shape[0]):
        original_img_name = original_img[i * test_loader.batch_size + j]
        slice_id = original_img_name.split('_')[-1].split('.')[0]
        output_name = f"{patient_id}_mask_{slice_id}.png"
        output_p = os.path.join(path_output, output_name)
        plt.imsave(output_p, mask_prediction[j,0,:,:], cmap='gray', format='png')
        plt.close()


# Post-processing for the pseudo labels
pseudo_imgs = []
pseudo_names = []

for file in sorted(os.listdir(path_output)):
    if file.endswith('.png'):
        p = imread(f"{path_output}/{file}", as_gray=True)
        pseudo_names.append(file)
        threshold = threshold_otsu(p)
        binary_manual = p > threshold
        pseudo_imgs.append(binary_manual)

pseudo_img = np.stack(np.array(pseudo_imgs), axis=-1)

post_dust = cc3d.dust(
            pseudo_img, threshold=500, connectivity=26, in_place=False
        )
post_clean_holes = binary_fill_holes(post_dust)

for i in range(pseudo_img.shape[2]):
    output_name_post = os.path.join(path_output_post, pseudo_names[i])
    plt.imsave(output_name_post, post_clean_holes[:, :, i], cmap='gray', format='png')
    plt.close()


# Calculate DICE coefficients
DICE_no_post = [] 
DICE_post = []    
manual_imgs = [imread(os.path.join(path_manual, file), as_gray=True) for file in manual_img]

for i, file in enumerate(manual_img):
    prediction = imread(os.path.join(path_output, file), as_gray=True)
    prediction_post = imread(os.path.join(path_output_post, file), as_gray=True)
    gt = manual_imgs[i]
    
    dice_no_post = dice_coefficient(gt, prediction)
    dice_post = dice_coefficient(gt, prediction_post)
    
    DICE_no_post.append(dice_no_post)
    DICE_post.append(dice_post)

print("DICE without post-processing:", np.mean(DICE_no_post))
print("DICE with post-processing:", np.mean(DICE_post))
