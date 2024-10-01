from skimage.io import imread
from skimage.filters import threshold_otsu
import numpy as np
import os
import re
from torchvision import transforms
from skimage import io
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
    binary_mask1 = y_true > 0
    binary_mask2 = y_pred > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = binary_mask1.sum() + binary_mask2.sum()
    
    # Calculate Dice score
    dice = 2 * intersection / union
    return dice


# Extract last number from a string
def extract_last_number(filename):
    matches = re.findall(r"\d+", filename)
    return int(matches[-1]) if matches else 0

transform = transforms.Compose([
    transforms.ToTensor()
])

mean_dice_ori = []
mean_dice_post = []

for i in range(1, 10):
    # Read images and create a 3D array
    real_imgs = []
    pred_imgs = []
    
    fold_mask = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/tifvolume/mask/0{i}.tif"
    p_prediction = f"/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/result_com_final/fold{i}/swintr/0{i}.tif"

    real_imgs = io.imread(fold_mask)
    pred_imgs = io.imread(p_prediction)

    post_dust = cc3d.dust(
            pred_imgs, threshold=500, connectivity=26, in_place=False
        )
    post_clean_holes = binary_fill_holes(post_dust)

    dice_scores_2d = []
    dice_scores_2d.append(dice_coefficient(real_imgs, pred_imgs))
    mean_dice_ori.append(np.mean(dice_scores_2d))

    dice_scores_2d_clean = []
    dice_scores_2d_clean.append(
            dice_coefficient(real_imgs, post_dust)
        )

    dice_scores_2d_clean_holes = []
    dice_scores_2d_clean_holes.append(
            dice_coefficient(real_imgs, post_clean_holes)
        )
    mean_dice_post.append(np.mean(dice_scores_2d_clean_holes))

    print(f"CV{i}: DICE without post-processing:", np.mean(dice_scores_2d))
    # print(f"       DICE post-processing (dust):", np.mean(dice_scores_2d_clean))
    # print(f"       DICE post-processing (fill holes):", np.mean(dice_scores_2d_clean_holes))

print(f"Mean DICE without post-processing:", np.mean(mean_dice_ori))
print("no, std", np.std(mean_dice_ori))
print(f"Mean DICE post-processing:", np.mean(mean_dice_post))
print("post, std", np.std(mean_dice_post))
