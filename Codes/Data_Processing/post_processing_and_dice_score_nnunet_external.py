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
from matplotlib import pyplot as plt


def remove_floating_objects(volume):
    struct = generate_binary_structure(3, 3)
    labeled_volume, _ = label(volume, structure=struct)
    largest_component_label = np.argmax(np.bincount(labeled_volume.flat)[1:]) + 1
    return labeled_volume == largest_component_label


# DICE
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection) / (union + 1e-8)


def extract_last_number(filename):
    matches = re.findall(r"\d+", filename)
    return int(matches[-1]) if matches else 0


def calculate_dice_scores(model_name):
    base_mask_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/External Dataset/{}/Manual Mask"
    base_pred_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_results/Dataset{}_Skull/{}__nnUNetPlans__2d/predictions"

    folds = [501, 502, 503, 504] #range(501, 502)
    mean_dice_ori, mean_dice_post = [], []
    patient_ids =  ["P13", "P16", "P30", "P38"] #["P13", "P16", "P30", "P38"]

    for fold_id, patient_id in zip(folds, patient_ids):
        mask_path = base_mask_path.format(patient_id)
        pred_path = base_pred_path.format(fold_id, model_name)

        print("Predictions path: ", pred_path)
        print("Masks path: ", mask_path)

        real_imgs = []
        pred_imgs = []
        for file in sorted(os.listdir(mask_path)):
            if file.endswith('.png'):
                manual = imread(os.path.join(mask_path, file), as_gray=True)
                threshold1 = threshold_otsu(manual)
                binary_manual = manual > threshold1
                real_imgs.append(binary_manual)
        real_imgs = np.stack(np.array(real_imgs), axis=-1)

        # Read and stack predicted images
        for file in sorted(os.listdir(pred_path)):
            if file.endswith('.png'):
                manual = imread(os.path.join(pred_path, file), as_gray=True)
                threshold1 = threshold_otsu(manual)
                binary_manual = manual > threshold1
                pred_imgs.append(binary_manual)
        pred_imgs = np.stack(np.array(pred_imgs), axis=-1)

        # Post-process predictions
        post_dust = cc3d.dust(pred_imgs, threshold=500, connectivity=26, in_place=False)
        post_clean_holes = binary_fill_holes(post_dust)

        # Calculate DICE scores for original and post-processed predictions
        dice_scores_ori = [dice_coefficient(real_imgs[:, :, x], pred_imgs[:, :, x]) for x in range(real_imgs.shape[2])]
        mean_dice_ori.append(np.mean(dice_scores_ori))

        dice_scores_post = [dice_coefficient(real_imgs[:, :, x], post_clean_holes[:, :, x]) for x in range(real_imgs.shape[2])]
        mean_dice_post.append(np.mean(dice_scores_post))

        print(f"FOLD {fold_id}: DICE without post-processing: {np.mean(dice_scores_ori)}")

        # # Plot 5 pairs of real and predicted images
        # for i in range(5):
        #     print(f"Image {i}")
        #     print(f"DICE without post-processing: {dice_scores_ori[i]}")
        #     print(f"DICE post-processing: {dice_scores_post[i]}")
        #     print()
        #     fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        #     ax[0].imshow(real_imgs[:, :, i], cmap='gray')
        #     ax[0].set_title('Real mask')
        #     ax[1].imshow(pred_imgs[:, :, i], cmap='gray')
        #     ax[1].set_title('Predicted mask')
        #     # save img to one folder above predict path using plt.savefig
        #     plt.savefig(os.path.join(pred_path, f"example_{i}.png"))
        #     plt.show()

        #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #     ax[0].imshow(real_imgs[:, :, i], cmap='gray')
        #     ax[0].set_title('Real')
        #     ax[1].imshow(post_clean_holes[:, :, i], cmap='gray')
        #     ax[1].set_title('Predicted (Post-processed)')
        #     plt.savefig(os.path.join(pred_path, f"example_{i}_post.png"))
        #     plt.show()

 


    # Calculate mean and standard deviation across all folds
    print(f"Mean DICE without post-processing: {np.mean(mean_dice_ori)*100}, std: {np.std(mean_dice_ori)*100}")
    print(f"Mean DICE post-processing: {np.mean(mean_dice_post)*100}, std: {np.std(mean_dice_post)*100}")

# Example usage:
if __name__ == "__main__":
    calculate_dice_scores("nnUNetTrainerSwinUNETR")