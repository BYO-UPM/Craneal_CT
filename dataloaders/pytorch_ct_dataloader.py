import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomAffine,
)


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
        for patient_id in os.listdir(root_dir):
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
        self.geometric_aug_list = [
            # Geometrics transformations
            RandomHorizontalFlip(p=1.0),  # Always apply horizontal flip
            RandomVerticalFlip(p=1.0),    # Always apply vertical flip
            RandomAffine(
                degrees=0, scale=(0.85, 0.99)
            ),  # Scaling only in x-axis
            RandomAffine(degrees=(-180,180)), # rotation +-180
        ]

        self.noise_Gaussian_aug = GaussianNoise(mean=0, std=0.1)
        self.noise_SaltPepper_aug = SaltAndPepperNoise(salt_prob=0.01, pepper_prob=0.01)
        self.noise_Normal_aug = LaplaceNoise(mean=0, std=0.1)
        self.noise_Poisson_aug = PoissonNoise()

    def __call__(self, image, mask):
        augmented_images = []
        augmented_masks = []
        # Apply geometric transformations
        #image = self.geometric_augmentations(image)
        #mask = self.geometric_augmentations(mask)
        #image, mask = self.geometric_augmentations(image, mask)

        for geom_aug in self.geometric_aug_list:
            transformed_image = geom_aug(image)
            transformed_mask = geom_aug(mask)

            # Apply a random noise augmentation to the transformed image
            noise_functions = [
                self.noise_Gaussian_aug,
                self.noise_SaltPepper_aug,
                self.noise_Normal_aug,
                self.noise_Poisson_aug
            ]
            noise_function = random.choice(noise_functions)
            final_image = noise_function(transformed_image)

            # Add the final augmented image and mask to the lists
            augmented_images.append(final_image)
            augmented_masks.append(transformed_mask)

        # Return the list of augmented images and masks
        return augmented_images, augmented_masks
        #return image, mask


class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Add same Gaussian noise to both image and mask with probability
        noise = torch.distributions.normal.Normal(self.mean, self.std).sample(
            image.size()
        )
        prob = random.random()
        if prob <= 0.2:
            noisy_image = image + noise
        else:
            noisy_image = image
        
        return noisy_image


class SaltAndPepperNoise:
    def __init__(self, salt_prob, pepper_prob):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, image):
        # Add salt and pepper noise to both image and mask
        prob = random.random()
        if prob <= 0.2:
            salt = torch.rand(image.size()) < self.salt_prob
            pepper = torch.rand(image.size()) < self.pepper_prob
            noisy_image = image.clone()
            noisy_image[salt] = 1
            noisy_image[pepper] = 0
        else:
            noisy_image = image
        return noisy_image


class LaplaceNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Add Laplace noise to both image and mask
        prob = random.random()
        if prob<= 0.2:
            noise = torch.distributions.laplace.Laplace(self.mean, self.std).sample(
                image.size()
            )
            noisy_image = image + noise
        else:
            noisy_image = image
        return noisy_image


class PoissonNoise:
    def __call__(self, image):
        # Add Poisson noise to both image and mask
        prob = random.random()
        if prob <= 0.2:
            noisy_image = torch.distributions.poisson.Poisson(image).sample()
        else:
            noisy_image = image
        return noisy_image
    

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, augmentation=None):
        self.subset = subset
        self.augmentation = augmentation
    
    def __len__(self):
        # 4 images augmentation + 1 original
        return len(self.subset) * 5
 
    def __getitem__(self, index):
        # Index for original image
        original_index = index // 5
        # image augmentation
        augmentation_index = index % 5

        original_img, mask_img, patient_id, slice_number = self.subset[index]
        
        if self.augmentation:
            # image and mask lists augmentation
            augmented_images, augmented_masks = self.augmentation(original_img, mask_img)
            # based on augmentation_index selecting pair of image and mask
            augmented_img = augmented_images[augmentation_index - 1]
            augmented_mask = augmented_masks[augmentation_index - 1]
        else:
            # no augmentation
            augmented_img = original_img
            augmented_mask = mask_img
        return augmented_img, augmented_mask, patient_id, slice_number
 
    def __len__(self):
        return len(self.subset)
