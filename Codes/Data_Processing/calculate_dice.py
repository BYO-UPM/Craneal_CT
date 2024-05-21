from skimage.io import imread
from skimage.filters import threshold_otsu
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import plotly.graph_objects as go
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


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


folds = ["fold1_P16", "fold2_P13", "fold3_P30", "fold4_P38", "fold5_P01"]
best_dice_score = []
old_dice_score = []

for fold in folds:
    print("============================= Predictions for fold: ", fold)

    path_groundtruth = (
        "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/post_process/"
        + fold
        + "/Manual Mask"
    )
    path_prediction = (
        "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/post_process/"
        + fold
        + "/Prediction"
    )

    pred_mask = os.listdir(path_prediction)
    pred_mask = sorted(pred_mask, key=extract_last_number)
    gt_mask = os.listdir(path_groundtruth)
    gt_mask = sorted(gt_mask, key=extract_last_number)
    pred_mask = [file for file in pred_mask if file.endswith(".png")]
    gt_mask = [file for file in gt_mask if file.endswith(".png")]

    # Read images and create a 3D array
    real_imgs = []
    pred_imgs = []

    for pred_file, gt_file in zip(pred_mask, gt_mask):
        real = imread(os.path.join(path_groundtruth, gt_file), as_gray=True)
        pred = imread(os.path.join(path_prediction, pred_file), as_gray=True)

        threshold1 = threshold_otsu(real)
        threshold2 = threshold_otsu(pred)

        binary_image1 = real > threshold1
        binary_image2 = pred > threshold2

        real_imgs.append(binary_image1)
        pred_imgs.append(binary_image2)

    real_imgs = np.stack(np.array(real_imgs), axis=-1)
    pred_imgs = np.stack(np.array(pred_imgs), axis=-1)

    # Plot 2D comparison of the 20th slice
    slice_index = 19  # 20th slice (0-based index)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(real_imgs[:, :, slice_index], cmap="gray")
    plt.title("Ground Truth - Slice 20")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_imgs[:, :, slice_index], cmap="gray")
    plt.title("Prediction - Slice 20")

    plt.tight_layout()
    plt.show()

    # Create interactive 3D plot
    def plot_3d_interactive(real_imgs, pred_imgs):
        verts_real, faces_real, _, _ = marching_cubes(real_imgs, level=0)
        verts_pred, faces_pred, _, _ = marching_cubes(pred_imgs, level=0)

        fig = go.Figure()

        # Ground truth mesh
        fig.add_trace(
            go.Mesh3d(
                x=verts_real[:, 0],
                y=verts_real[:, 1],
                z=verts_real[:, 2],
                i=faces_real[:, 0],
                j=faces_real[:, 1],
                k=faces_real[:, 2],
                color="yellow",
                opacity=0.8,
                name="Ground Truth",
            )
        )

        # Prediction mesh
        fig.add_trace(
            go.Mesh3d(
                x=verts_pred[:, 0],
                y=verts_pred[:, 1],
                z=verts_pred[:, 2],
                i=faces_pred[:, 0],
                j=faces_pred[:, 1],
                k=faces_pred[:, 2],
                color="blue",
                opacity=0.1,
                name="Prediction",
            )
        )

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title="3D Ground Truth and Prediction Comparison",
        )

        fig.show()

    # plot_3d_interactive(real_imgs, pred_imgs)

    # Create a GIF of the 3D reconstruction overlay
    def create_gif(real_imgs, pred_imgs, gif_name):
        images = []
        for angle in range(0, 360, 10):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Plot ground truth
            verts, faces, _, _ = marching_cubes(real_imgs, level=0)
            mesh = Poly3DCollection(verts[faces], alpha=0.3, color="yellow")
            ax.add_collection3d(mesh)

            # Plot prediction
            verts, faces, _, _ = marching_cubes(pred_imgs, level=0)
            mesh = Poly3DCollection(verts[faces], alpha=0.3, color="blue")
            ax.add_collection3d(mesh)

            ax.view_init(30, angle)
            ax.set_xlim(0, real_imgs.shape[0])
            ax.set_ylim(0, real_imgs.shape[1])
            ax.set_zlim(0, real_imgs.shape[2])

            # Save to a temporary file
            filename = f"_tmp{angle}.png"
            plt.savefig(filename)
            images.append(imageio.imread(filename))
            plt.close(fig)

        # Create the GIF
        imageio.mimsave(gif_name, images, fps=10)

    # create_gif(real_imgs, pred_imgs, "3d_comparison.gif")

    # Post-processing
    import cc3d

    pred_imgs_clean = cc3d.dust(
        pred_imgs, threshold=500, connectivity=26, in_place=False
    )

    pred_imgs_clean = cc3d.connected_components(pred_imgs_clean, connectivity=26)

    pred_imgs_clean = pred_imgs_clean > 0

    # Fill holes with scipy binary_fill_holes
    from scipy.ndimage import binary_fill_holes

    pred_imgs_clean_holes = binary_fill_holes(pred_imgs_clean)

    # Plot 2D comparison of the 20th slice
    slice_index = 10  # 20th slice (0-based index)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(real_imgs[:, :, slice_index], cmap="gray")
    plt.title("Ground Truth - Slice 20")

    plt.subplot(2, 2, 2)
    plt.imshow(pred_imgs[:, :, slice_index], cmap="gray")
    plt.title("Prediction - Slice 20")

    plt.subplot(2, 2, 3)
    plt.imshow(pred_imgs_clean[:, :, slice_index], cmap="gray")
    plt.title("Prediction Cleaned - Slice 20")

    plt.subplot(2, 2, 4)
    plt.imshow(pred_imgs_clean_holes[:, :, slice_index], cmap="gray")
    plt.title("Prediction Cleaned with Holes - Slice 20")

    plt.tight_layout()
    plt.show()

    # Comparison of dice scores, before and after postprocess
    dice_scores = dice_coefficient(real_imgs, pred_imgs)
    dice_scores_clean = dice_coefficient(real_imgs, pred_imgs_clean)
    dice_score_clean_holes = dice_coefficient(real_imgs, pred_imgs_clean_holes)

    # print comparison
    print("Dice score before post-processing:", dice_scores)
    print("Dice score after post-processing:", dice_scores_clean)
    print("Dice score after post-processing with holes:", dice_score_clean_holes)

    # Calculate dice scores in 2D
    dice_scores_2d = []
    for i in range(real_imgs.shape[2]):
        dice_scores_2d.append(dice_coefficient(real_imgs[:, :, i], pred_imgs[:, :, i]))

    dice_scores_2d_clean = []
    for i in range(real_imgs.shape[2]):
        dice_scores_2d_clean.append(
            dice_coefficient(real_imgs[:, :, i], pred_imgs_clean[:, :, i])
        )

    dice_scores_2d_clean_holes = []
    for i in range(real_imgs.shape[2]):
        dice_scores_2d_clean_holes.append(
            dice_coefficient(real_imgs[:, :, i], pred_imgs_clean_holes[:, :, i])
        )

    dice_scores_2d = np.mean(dice_scores_2d)
    dice_scores_2d_clean = np.mean(dice_scores_2d_clean)
    dice_scores_2d_clean_holes = np.mean(dice_scores_2d_clean_holes)
    # print dice scores
    print("Dice scores 2D before post-processing:", dice_scores_2d)
    print("Dice scores 2D after post-processing:", dice_scores_2d_clean)
    print(
        "Dice scores 2D after post-processing with holes:", dice_scores_2d_clean_holes
    )

    old_dice_score.append(dice_scores_2d)
    best_dice_score.append(dice_scores_2d_clean_holes)

print("Old dice score: ", old_dice_score)
print("Best dice score: ", best_dice_score)
print("Mean old dice score: ", np.mean(old_dice_score))
print("Mean best dice score: ", np.mean(best_dice_score))
print("Standard deviation old dice score: ", np.std(old_dice_score))
print("Standard deviation best dice score: ", np.std(best_dice_score))
print("Mean improvement: ", np.mean(best_dice_score) - np.mean(old_dice_score))
