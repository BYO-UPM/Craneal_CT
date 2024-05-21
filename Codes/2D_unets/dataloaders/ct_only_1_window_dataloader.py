import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CATScansDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentations=None):
        """
        Args:
            root_dir (string): Directory with all the CT scans, structured as described.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations
        self.samples = []
        self.patient_id = [] 

        # Walk through the root directory and create a list of (image, mask) pairs
        for patient_id in sorted(os.listdir(root_dir)):
            patient_dir = os.path.join(root_dir, patient_id)

            original_dir = os.path.join(patient_dir, "Original")
            mask_dir = os.path.join(patient_dir, "Mask")

            # Use walk to avoid .ds_store fles
            for root, _, files in os.walk(original_dir):
                for original_file in files:
                    self.patient_id.append(patient_id)
                    slice_number = original_file.split("_")[
                        -1
                    ]  # Assuming file format is consistent
                    mask_file = original_file.split("_")[0] + "_mask_" + slice_number
                    self.samples.append(
                        (
                            os.path.join(original_dir, original_file),
                            os.path.join(mask_dir, mask_file),
                            patient_id,
                            slice_number,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_path, mask_path, patient_id, slice_number = self.samples[idx]
        original_image = Image.open(original_path).convert("L")  # Convert to grayscale
        mask_image = Image.open(mask_path).convert("L")

        if self.transform:
            original_image = self.transform(original_image) / 255
            mask_image = self.transform(mask_image) / 255
            # Binarise the mask
            mask_image = (mask_image > 0.5).float()

        if self.augmentations:
            original_image, mask_image = self.augmentations(original_image, mask_image)

        return (
            original_image,
            mask_image,
            patient_id,
            slice_number,
        )


# Create data augmentation and transformation pipeline
class CustomAugmentation:
    def __init__(self):
        self.window1 = windowSetup(window_center=330, window_width=350)
        self.window2 = windowSetup(window_center=-30, window_width=30)
        self.window3 = windowSetup(window_center=-100, window_width=900)
        self.window4 = windowSetup(window_center=25, window_width=95)
        self.window5 = windowSetup(window_center=300, window_width=2500)
        self.window6 = windowSetup(window_center=10, window_width=450)


    def __call__(self, image, mask):
        augmented_images = []
        augmented_masks = []

        augmented_images.append(image)
        augmented_masks.append(mask)

        window_ori = self.window6(image)
        augmented_images.append(window_ori)
        augmented_masks.append(mask)

        # Return the list of augmented images and masks
        return augmented_images, augmented_masks
    

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, augmentation=None):
        self.subset = subset
        self.augmentation = augmentation
    
    def __len__(self):
        # 4 images augmentation + 6 windows 
        return len(self.subset) * 2 
 
    def __getitem__(self, index):
        # Calculate original index and augmentation index
        original_idx = index // 2
        aug_idx = index % 2
        original_img, mask_img, patient_id, slice_number = self.subset[original_idx]
        
        if self.augmentation:
            transformed_images, transformed_masks = self.augmentation(original_img,
                                                                      mask_img)
            original_img = transformed_images[aug_idx]
            mask_img = transformed_masks[aug_idx]
        else:
            original_img = original_img  # No transformation
       
        return original_img, mask_img, patient_id, slice_number

class windowSetup:
    def __init__(self, window_center, window_width):
        self.window_center = window_center
        self.window_width = window_width

    def __call__(self, image):
        # Apply windowing to the image
        image = self.windowing(image, self.window_center, self.window_width)
        return image

    def windowing(self, image, window_center, window_width):
        # Apply windowing to the image
        window_center = (window_center + 1024) / (3071 + 1024)
        window_width = window_width / (3071 + 1024)
        lower_bound = window_center - window_width / 2
        upper_bound = window_center + window_width / 2
        image = torch.clamp(image, min=lower_bound, max=upper_bound)
        image = (image - lower_bound) / (upper_bound - lower_bound)

        return image
