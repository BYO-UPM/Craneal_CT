import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import cv2

class CATScansDataset(Dataset):
    def __init__(self, root_dir, transform=None, window=None):
        """
        Args:
            root_dir (string): Directory with all the CT scans, structured as described.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.window = window
        self.filenames = [f for f in sorted(os.listdir(root_dir)) if f.endswith('.png')]
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        #image = Image.open(img_path).convert("L") # Convert to grayscale
        if self.transform:
            normal = transforms.Normalize(mean=0, std=(1 / 255))
            numpy_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
            equalized_image = clahe.apply(numpy_image)  # Apply CLAHE
            original_image = self.transform(equalized_image) / 255
            original_image = normal(original_image)

        if self.window:
            image = self.window(image)
        return image


class PreprocessWindow:
    def __init__(self):
        self.window = windowSetup(window_center=500, window_width=2000)

    def __call__(self, image):
        window_ori = self.window(image)
        return window_ori


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
        #window_center = (window_center + 1024) / (3071 + 1024)
        #window_width = window_width / (3071 + 1024)
        lower_bound = window_center - window_width / 2
        upper_bound = window_center + window_width / 2
        image = torch.clamp(image, min=lower_bound, max=upper_bound)
        image = 255*(image - lower_bound) / (upper_bound - lower_bound)

        return image
