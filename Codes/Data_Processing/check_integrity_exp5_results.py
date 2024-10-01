import os
import cv2
import numpy as np
import hashlib
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
from tqdm import tqdm

# Directories (replace with your actual paths)
train_images_dir = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset501_Skull/imagesTr'
train_labels_dir = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset501_Skull/labelsTr'
test_images_dir = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset501_Skull/imagesTs'
test_labels_dir = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/Dataset/Labeled Data PNG/External Dataset/P13/Manual Mask'

def load_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            filepath = os.path.join(directory, filename)
            image_paths.append(filepath)
    return image_paths

train_image_paths = load_image_paths(train_images_dir)
test_image_paths = load_image_paths(test_images_dir)

# Build hashes for train images
train_hashes = {}
for path in tqdm(train_image_paths, desc='Hashing training images'):
    img_hash = imagehash.phash(Image.open(path))
    train_hashes[path] = img_hash

# Compare test images with train images
duplicates = []
for test_path in tqdm(test_image_paths, desc='Comparing test images'):
    test_img = Image.open(test_path)
    test_hash = imagehash.phash(test_img)
    for train_path, train_hash in train_hashes.items():
        hash_diff = test_hash - train_hash
        if hash_diff < 5:  # Adjust threshold as needed
            # Further verify using SSIM
            img1 = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            img1_resized = cv2.resize(img1, (256, 256))
            img2_resized = cv2.resize(img2, (256, 256))
            ssim_index = ssim(img1_resized, img2_resized)
            if ssim_index > 0.9:  # Adjust threshold as needed
                duplicates.append((train_path, test_path, ssim_index))
                print(f"Duplicate found: {train_path} and {test_path} with SSIM: {ssim_index}")

if not duplicates:
    print("No similar images found between training and testing sets.")
else:
    print(f"Found {len(duplicates)} similar images.")


def load_images_with_hashes(directory):
    image_hashes = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.nii', '.nii.gz')):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Compute a hash of the image data
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                image_hashes[img_hash] = filename
            else:
                print(f"Warning: Could not read image {filepath}")
    return image_hashes

# Load images with hashes
print("Loading and hashing training images...")
train_image_hashes = load_images_with_hashes(train_images_dir)
print("Loading and hashing testing images...")
test_image_hashes = load_images_with_hashes(test_images_dir)

# Find duplicates by comparing hashes
print("\nComparing image hashes...")
duplicate_images = set(train_image_hashes.keys()) & set(test_image_hashes.keys())
if duplicate_images:
    for img_hash in duplicate_images:
        print(f"Duplicate image found: {train_image_hashes[img_hash]} (train) and {test_image_hashes[img_hash]} (test)")
else:
    print("No duplicate images found between training and testing sets.")

# Repeat for labels
print("\nLoading and hashing training labels...")
train_label_hashes = load_images_with_hashes(train_labels_dir)
print("Loading and hashing testing labels...")
test_label_hashes = load_images_with_hashes(test_labels_dir)

print("\nComparing label hashes...")
duplicate_labels = set(train_label_hashes.keys()) & set(test_label_hashes.keys())
if duplicate_labels:
    for lbl_hash in duplicate_labels:
        print(f"Duplicate label found: {train_label_hashes[lbl_hash]} (train) and {test_label_hashes[lbl_hash]} (test)")
else:
    print("No duplicate labels found between training and testing sets.")
