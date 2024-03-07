import os
import torch
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
        self.geometric_augmentations = Compose(
            [
                # Geometrics transformations
                RandomHorizontalFlip(p=1.0),  # Always apply horizontal flip
                RandomVerticalFlip(p=1.0),  # Always apply vertical flip
                RandomAffine(
                    degrees=0, translate=(0.15, 0.15)
                ),  # Scaling in x-axis and y-axis
            ]
        )

        self.noise_augmentations = Compose(
            [
                GaussianNoise(mean=0, std=0.1),
                SaltAndPepperNoise(salt_prob=0.01, pepper_prob=0.01),
                LaplaceNoise(mean=0, std=0.1),
                PoissonNoise(),
            ]
        )

    def __call__(self, image, mask):
        # Apply geometric transformations
        image, mask = self.geometric_augmentations(image, mask)

        # Apply noise augmentations separately
        image = self.noise_augmentations(image)

        return image, mask


class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Add same Gaussian noise to both image and mask
        noise = torch.distributions.normal.Normal(self.mean, self.std).sample(
            image.size()
        )
        noisy_image = image + noise
        return noisy_image


class SaltAndPepperNoise:
    def __init__(self, salt_prob, pepper_prob):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, image):
        # Add salt and pepper noise to both image and mask
        salt = torch.rand(image.size()) < self.salt_prob
        pepper = torch.rand(image.size()) < self.pepper_prob
        noisy_image = image.clone()
        noisy_image[salt] = 1
        noisy_image[pepper] = 0
        return noisy_image


class LaplaceNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        # Add Laplace noise to both image and mask
        noise = torch.distributions.laplace.Laplace(self.mean, self.std).sample(
            image.size()
        )
        noisy_image = image + noise
        return noisy_image


class PoissonNoise:
    def __call__(self, image, mask):
        # Add Poisson noise to both image and mask
        noisy_image = torch.distributions.poisson.Poisson(image).sample()
        return noisy_image
