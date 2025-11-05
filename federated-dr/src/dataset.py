"""
DATASET MODULE - Data Loading & Preprocessing

This module handles:
- Loading Diabetic Retinopathy images
- Filtering available images
- Image transformations and augmentations
- Dataset splitting utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import os


class DiabeticRetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy images
    
    Handles APTOS-style datasets with CSV labels and image directories.
    Automatically filters to only include images that actually exist.
    """

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            csv_file: Path to CSV with columns: id_code, diagnosis
            img_dir: Directory containing images (PNG or JPEG)
            transform: torchvision transforms to apply
        """
        # Read CSV
        df = pd.read_csv(csv_file)

        # FILTER: Only keep images that actually exist
        print("Filtering CSV to only include available images...")
        available_images = []
        available_labels = []

        for idx, row in df.iterrows():
            img_name = row['id_code']
            label = row['diagnosis']

            # Check if image exists (try both extensions)
            img_path_png = os.path.join(img_dir, f"{img_name}.png")
            img_path_jpeg = os.path.join(img_dir, f"{img_name}.jpeg")

            if os.path.exists(img_path_png) or os.path.exists(img_path_jpeg):
                available_images.append(img_name)
                available_labels.append(label)

        # Create filtered dataframe
        self.labels_df = pd.DataFrame({
            'id_code': available_images,
            'diagnosis': available_labels
        })

        self.img_dir = img_dir
        self.transform = transform

        # Print dataset statistics
        print("Loaded {} images (filtered from {} in CSV)".format(
            len(self.labels_df), len(df)))
        print("Class distribution: {}".format(Counter(self.labels_df['diagnosis'])))

    def __len__(self):
        """Returns total number of samples"""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Loads and returns one sample
        
        Args:
            idx: Index of sample
            
        Returns:
            image: Transformed PIL image as tensor
            label: Diagnosis label (0-4)
        """
        # Get image name and label
        img_name = self.labels_df.iloc[idx]['id_code']
        label = self.labels_df.iloc[idx]['diagnosis']

        # Load image (try .png first, then .jpeg)
        img_path = os.path.join(self.img_dir, "{}.png".format(img_name))
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, "{}.jpeg".format(img_name))

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("Error loading {}: {}".format(img_path, e))
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode='train', img_size=224):
    """
    Get image transformations for train/val/test
    
    Args:
        mode: 'train', 'val', or 'test'
        img_size: Target image size (default 224)
        
    Returns:
        transforms.Compose: Image transformations
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

