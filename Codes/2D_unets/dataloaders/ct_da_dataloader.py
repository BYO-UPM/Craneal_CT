import os
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import math
import torch.nn.functional as F
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomAffine,
    GaussianBlur,
)

# Dataloader for fix training set
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
                    if original_file.endswith(".png"):
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
        mask_image = Image.open(mask_path).convert("L")

        if self.transform:
            normal = transforms.Normalize(mean=0, std=(1 / 255))

            numpy_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
            equalized_image = clahe.apply(numpy_image)  # Apply CLAHE
            original_image = self.transform(equalized_image) / 255
            original_image = normal(original_image)
            
            mask_image = self.transform(mask_image) / 255
            mask_image = normal(mask_image)
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


# Dataloader for subset 3 (random CT scans)
class RandomScansDataset(Dataset):
    def __init__(self, root_dir, patient_idx, transform=None, augmentations=None):

        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations
        self.samples = []
        self.patient_idx = patient_idx

        # Walk through the root directory and create a list of (image, mask) pairs
        for patient_id in patient_idx:
            patient_dir = os.path.join(root_dir, patient_id)

            original_dir = os.path.join(patient_dir, "Original")
            mask_dir = os.path.join(patient_dir, "Pseudo Labels1") # Fold 1, 2, 3, 4

            # Use walk to avoid .ds_store fles
            for root, _, files in os.walk(original_dir):
                for original_file in files:
                    if original_file.endswith(".png"):
                        slice_number = original_file.split("_")[
                            -1
                        ]  # Assuming file format is consistent
                        mask_file = original_file.split("_")[0] + "_pseudo_" + slice_number
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
        mask_image = Image.open(mask_path).convert("L")

        if self.transform:
            normal = transforms.Normalize(mean=0, std=(1 / 255))
            numpy_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
            equalized_image = clahe.apply(numpy_image)  # Apply CLAHE
            original_image = self.transform(equalized_image) / 255
            original_image = normal(original_image)
            
            mask_image = self.transform(mask_image) / 255
            mask_image = normal(mask_image)
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
        self.scale_x = ScaleX()
        self.degree = random.randint(-180, 180)
        self.noise_Gaussian_aug = GaussianNoise(mean=0, std=0.1)
        self.noise_SaltPepper_aug = SaltAndPepperNoise(salt_prob=0.01, pepper_prob=0.01)
        self.noise_Normal_aug = LaplaceNoise(mean=0, std=0.1)
        self.noise_Poisson_aug = PoissonNoise()
        self.gaussian_blur = GaussianBlur(kernel_size=(15, 15), sigma=(2.0, 3.0))
        self.circleCrop = circleCrop(radius=150)
        self.randomCrop = randomCrop(square_size=180, triangle_size=150, circle_size=100)
        self.window1 = windowSetup(window_center=330, window_width=350)
        self.window2 = windowSetup(window_center=-30, window_width=30)
        self.window3 = windowSetup(window_center=-100, window_width=900)
        self.window4 = windowSetup(window_center=25, window_width=95)
        self.window5 = windowSetup(window_center=300, window_width=2500)
        self.window6 = windowSetup(window_center=10, window_width=450)

        self.geometric_aug_list = [
            # Geometrics transformations
            RandomHorizontalFlip(p=1.0),  # Always apply horizontal flip
            RandomVerticalFlip(p=1.0),    # Always apply vertical flip
            self.scale_x,  # Scaling only in x-axis
            RandomAffine(degrees=(self.degree,self.degree)), # Rotation
        ]

        self.window_list = [
            # List of 6 new windows
            self.window1,
            self.window2,
            self.window3,
            self.window4,
            self.window5,
            self.window6,
        ]


    def __call__(self, image, mask):
        augmented_images = []
        augmented_masks = []

        augmented_images.append(image)
        augmented_masks.append(mask)

        blur_image = self.gaussian_blur(image)
        augmented_images.append(blur_image)
        augmented_masks.append(mask)

        circle_image = self.circleCrop(image)
        augmented_images.append(circle_image)
        circle_mask = self.circleCrop(mask)
        augmented_masks.append(circle_mask)
        
        randomt_image, randomt_mask = self.randomCrop(image, mask)
        augmented_images.append(randomt_image)
        augmented_masks.append(randomt_mask)

        for window_aug in self.window_list:
            # Apply windowing to the image
            window_ori = window_aug(image)
            # Add the final augmented image and mask to the lists
            augmented_images.append(window_ori)
            augmented_masks.append(mask)
        
        for geom_aug in self.geometric_aug_list:
            # Apply geometric transformations to the image and mask
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

            for window_aug in self.window_list:
                # Apply windowing to the image
                window_image = window_aug(transformed_image)
                # Add the final augmented image and mask to the lists
                augmented_images.append(window_image)
                augmented_masks.append(transformed_mask)

        # Return the list of augmented images and masks
        return augmented_images, augmented_masks
    

class ScaleX:
    def __call__(self, image):

        _, H, W = image.size()
        scaled_W = int(W * 0.85)

        # Use grid_sample for scaling
        # Create affine matrix for scaling
        theta = torch.tensor([
            [0.85, 0, 0],
            [0, 1, 0],
        ], dtype=torch.float).unsqueeze(0)
        
        grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
        scaled_x = F.grid_sample(image.unsqueeze(0), grid, align_corners=False)

        return scaled_x.squeeze(0)


class circleCrop:
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        # Crop the image to a circle with radius
        _, w, h = image.size()
        x, y = 150, h // 2
        mask = torch.zeros((w, h))
        for i in range(w):
            for j in range(h):
                if (i - x) ** 2 + (j - y) ** 2 <= self.radius ** 2:
                    mask[i, j] = 1
        masked_image = image * mask

        roi = masked_image[:, 0:300, 106:406]
        # Calculate the scale factor to expand the ROI to 512x512
        
        # Resize the ROI to 512x512
        resized_image = F.interpolate(roi.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
        resized_image = resized_image.squeeze(0)  # Remove the batch dimension
        
        return resized_image


class randomCrop:
    def __init__(self, square_size, triangle_size, circle_size):
        self.square_size = square_size
        self.triangle_size = triangle_size
        self.circle_size = circle_size
    
    def __call__(self, image, mask):
        # Image size
        _, w, h = image.size()
        canva = torch.zeros((w, h))

        points = []
        while len(points) < 3:
            x = random.randint(100, w-100)
            y = random.randint(100, h-100)
            point = (x, y)
            if all(abs(point[0] - p[0]) > 50 and abs(point[1] - p[1]) > 50 for p in points):
                points.append(point)
        
        # Draw the square
        canva[points[0][0]:points[0][0]+self.square_size, points[0][1]:points[0][1]+self.square_size] = 1

        # Draw the circle
        for i in range(w):
            for j in range(h):
                if (i - points[1][0]) ** 2 + (j - points[1][1]) ** 2 <= self.circle_size ** 2:
                    canva[i, j] = 1

        # Draw the triangle
        center_x, center_y = points[2]
        x1 = center_x - self.triangle_size // 2
        y1 = center_y + int(self.triangle_size * math.sqrt(3) / 2)
        x2 = center_x + self.triangle_size // 2
        y2 = center_y + int(self.triangle_size * math.sqrt(3) / 2)
        x3 = center_x
        y3 = center_y - int(self.triangle_size * math.sqrt(3) / 2)

        # Determine the range of x and y coordinates for the triangle
        min_x = min(x1, x2, x3)
        max_x = max(x1, x2, x3)
        min_y = min(y1, y2, y3)
        max_y = max(y1, y2, y3)

        # Limit the range within the canvas dimensions
        min_x = max(0, min_x)
        max_x = min(canva.shape[0] - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(canva.shape[1] - 1, max_y)

        # Draw triangle
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                if self.is_inside_triangle(i, j, x1, y1, x2, y2, x3, y3):
                    canva[i, j] = 1

        new_image = image * canva
        new_mask = mask * canva
        return new_image, new_mask
    
    def is_inside_triangle(self, x, y, x1, y1, x2, y2, x3, y3):
    # Check if point (x, y) is inside the triangle defined by (x1, y1), (x2, y2), (x3, y3)
        area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
        s = 1 / (2 * area) * (y1 * x3 - x1 * y3 + (y3 - y1) * x + (x1 - x3) * y)
        t = 1 / (2 * area) * (x1 * y2 - y1 * x2 + (y1 - y2) * x + (x2 - x1) * y)
        return 0 <= s <= 1 and 0 <= t <= 1 and s + t <= 1


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
        # 4 images augmentation + 6 windows + blur + circleCrop + randomCrop
        return len(self.subset) * 38 
 
    def __getitem__(self, index):
        # Calculate original index and augmentation index
        original_idx = index // 38
        aug_idx = index % 38
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
