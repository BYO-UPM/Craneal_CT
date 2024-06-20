from skimage.io import imread
from skimage.filters import threshold_otsu
import numpy as np
import os
import re
from torchvision import transforms
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
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

path_manual = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed"
#path = sorted(os.listdir(path_manual))
path = ['S090899',
 'S090934',
 'S090946',
 'S090947',
 'S090957',
 'S090967',
 'S090977',
 'S090979',
 'S090981']
path_pre = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/results_e3/AUFLVGG"

transform = transforms.Compose([
    transforms.ToTensor()
])

encoder = "vgg16"

mean_dice_ori = []
mean_dice_post = []

for i in range(1, 10):
    # Read images and create a 3D array
    real_imgs = []
    pred_imgs = []
    
    fold_mask = f"{path_manual}/{path[i-1]}/Mask"
    p_prediction = f"{path_pre}/cv{i}"
    fold_prediction = sorted(os.listdir(p_prediction), key=extract_last_number)

    for file in sorted(os.listdir(fold_mask)):
        if file.endswith('.png'):
            manual = imread(f"{fold_mask}/{file}", as_gray=True)
            threshold1 = threshold_otsu(manual)
            binary_manual = manual > threshold1
            real_imgs.append(binary_manual)
    real_imgs = np.stack(np.array(real_imgs), axis=-1)

    for file in fold_prediction:
        if file.endswith('.png'):
            predict = imread(f"{p_prediction}/{file}", as_gray=True)
            threshold2 = threshold_otsu(predict)
            binary_predict = predict > threshold2
            pred_imgs.append(binary_predict) 
    pred_imgs = np.stack(np.array(pred_imgs), axis=-1)

    post_dust = cc3d.dust(
            pred_imgs, threshold=500, connectivity=26, in_place=False
        )
    post_clean_holes = binary_fill_holes(post_dust)

    dice_scores_2d = []
    for x in range(real_imgs.shape[2]):
        dice_scores_2d.append(dice_coefficient(real_imgs[:, :, x], pred_imgs[:, :, x]))
    mean_dice_ori.append(np.mean(dice_scores_2d))

    dice_scores_2d_clean = []
    for x in range(real_imgs.shape[2]):
        dice_scores_2d_clean.append(
            dice_coefficient(real_imgs[:, :, x], post_dust[:, :, x])
        )

    dice_scores_2d_clean_holes = []
    for x in range(real_imgs.shape[2]):
        dice_scores_2d_clean_holes.append(
            dice_coefficient(real_imgs[:, :, x], post_clean_holes[:, :, x])
        )
    mean_dice_post.append(np.mean(dice_scores_2d_clean_holes))

    print(f"CV{i}: DICE of using {encoder} encoder without post-processing:", np.mean(dice_scores_2d))
    print(f"       DICE of using {encoder} encoder post-processing (dust):", np.mean(dice_scores_2d_clean))
    print(f"       DICE of using {encoder} encoder post-processing (fill holes):", np.mean(dice_scores_2d_clean_holes))

print(f"Mean DICE of using {encoder} encoder without post-processing:", np.mean(mean_dice_ori))
print("no, std", np.std(mean_dice_ori))
print(f"Mean DICE of using {encoder} encoder post-processing:", np.mean(mean_dice_post))
print("post, std", np.std(mean_dice_post))
