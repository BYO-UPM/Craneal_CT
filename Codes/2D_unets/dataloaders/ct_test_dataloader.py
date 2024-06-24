import os
from torch.utils.data import Dataset
from torchvision import transforms
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
        if self.transform:
            normal = transforms.Normalize(mean=0, std=(1 / 255))
            numpy_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
            equalized_image = clahe.apply(numpy_image)  # Apply CLAHE
            original_image = self.transform(equalized_image) / 255
            original_image = normal(original_image)

        return original_image
