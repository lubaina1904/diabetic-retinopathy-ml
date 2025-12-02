
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
import os


class DiabeticRetinopathyDataset(Dataset):
    
    def __init__(self, csv_file, img_dir, transform=None):
        
        df = pd.read_csv(csv_file)

        print("Filtering CSV to only include available images...")
        available_images = []
        available_labels = []

        for idx, row in df.iterrows():
            img_name = row['id_code']
            label = row['diagnosis']

            # Check if image exists 
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
        return len(self.labels_df)

    def __getitem__(self, idx):
        
        img_name = self.labels_df.iloc[idx]['id_code']
        label = self.labels_df.iloc[idx]['diagnosis']

        img_path = os.path.join(self.img_dir, "{}.png".format(img_name))
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, "{}.jpeg".format(img_name))

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("Error loading {}: {}".format(img_path, e))
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode='train', img_size=224):
    
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
    else:  
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

