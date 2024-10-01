import os
import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import cc3d
from scipy.ndimage import binary_fill_holes

# Paths to your images (replace these with your actual paths)
real_img_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset364_Skull/imagesTs/13_15_0000.png'
#/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/External Dataset/P16/Manual Mask/P16_mask_201.png
real_mask_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/External Dataset/P38/Manual Mask/P38_mask_087.png'
model1_mask_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_results/Dataset364_Skull/withDA/nnUNetTrainerSwinUNETR__nnUNetPlans__2d/predictions/13_15.png'
model2_mask_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_results/Dataset368_Skull/withDA/nnUNetTrainerSwinUNETR__nnUNetPlans__2d/predictions/13_15.png'
model3_mask_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_results/Dataset504_Skull/nnUNetTrainerSwinUNETR__nnUNetPlans__2d/predictions/13_15.png'

# List of image paths
image_paths = [
    real_img_path,
    real_mask_path,
    model1_mask_path,
    model2_mask_path,
    model3_mask_path
]
names_files = ["real_img", "real_mask", "model1_mask", "model2_mask", "model3_mask"]

# Directory to save processed images
output_dir = 'paper_images'
os.makedirs(output_dir, exist_ok=True)

def process_and_save_image(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Check if the image is a mask (assumed to be single-channel)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Check if the mask is in 0-1 range
        unique_values = np.unique(img)
        if set(unique_values).issubset({0, 1}):
            # Normalize mask to 0-255
            img = (img * 255).astype(np.uint8)
        else:
            # Ensure the mask is in uint8 format
            img = img.astype(np.uint8)
    else:
        # For real images, ensure it's in 8-bit format
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert image to RGB if it's in BGR format
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Convert grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Save image with dpi=300 using PIL
    pil_image = Image.fromarray(img)
    pil_image.save(output_path, dpi=(300, 300))

# Process and save each image
for image_path in image_paths:
    filename = os.path.basename(image_path)
    # Output path is output_directory + variable name + '.png'
    output_path = os.path.join(output_dir, names_files[image_paths.index(image_path)] + '.png')

    process_and_save_image(image_path, output_path)

# Now, apply post-processing to model3_mask.png and save as model3_finalpp_mask.png
model3_output_path = os.path.join(output_dir, "model3_finalpp_mask.png")
model3_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

if model3_image is None:
    print(f"Error: Could not read image at {model3_output_path}")
else:
    # Check if the mask is in 0-1 range and normalize if necessary
    unique_values = np.unique(model3_image)
    if set(unique_values).issubset({0, 1}):
        binary_mask = model3_image
    else:
        # Apply Otsu's threshold to binarize the mask
        threshold = threshold_otsu(model3_image)
        binary_mask = (model3_image > threshold).astype(np.uint8)

    # Expand dimensions to create a 3D volume (required by cc3d)
    binary_mask_3d = np.expand_dims(binary_mask, axis=0)

    # Fill holes in the binary mask
    post_clean_holes = binary_fill_holes(binary_mask_3d).astype(np.uint8)

    # Remove small connected components (threshold=500 pixels)
    post_dust = cc3d.dust(post_clean_holes, threshold=500, connectivity=26, in_place=False)

    # Collapse the 3D volume back to 2D
    post_processed_mask = post_clean_holes[0, :, :]

    # Normalize mask to 0-255
    post_processed_mask = (post_processed_mask * 255).astype(np.uint8)

    # Convert to RGB
    post_processed_mask_rgb = cv2.cvtColor(post_processed_mask, cv2.COLOR_GRAY2RGB)

    # Save the post-processed image
    post_processed_output_path = os.path.join(output_dir, 'model3_finalpp_mask.png')
    pil_image = Image.fromarray(post_processed_mask_rgb)
    pil_image.save(post_processed_output_path, dpi=(300, 300))

    print(f"Post-processed image saved as '{post_processed_output_path}' with dpi=300.")

print(f"All images have been processed and saved in '{output_dir}' with dpi=300.")
