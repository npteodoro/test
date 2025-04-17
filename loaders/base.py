import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base dataset class with common functionality for all dataset types"""
    
    def __init__(self, csv_file, root_dir, transform=None, split='train', split_ratio=[0.7, 0.15, 0.15], seed=42):
        """
        Args:
            csv_file: Path to CSV file with image paths and labels
            root_dir: Root directory for dataset
            transform: Transform function to apply to images and masks
            split: 'train', 'val', or 'test'
            split_ratio: Ratio for train/val/test splits
            seed: Random seed for reproducibility
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.rgb_path = os.path.join(root_dir, 'rgb')
        self.mask_path = os.path.join(root_dir, 'mask')
        self.transform = transform
        
        # Create train/val/test splits if needed
        if split in ['train', 'val', 'test']:
            np.random.seed(seed)
            indices = np.random.permutation(len(self.data_frame))
            train_end = int(split_ratio[0] * len(indices))
            val_end = train_end + int(split_ratio[1] * len(indices))
            
            if split == 'train':
                self.data_frame = self.data_frame.iloc[indices[:train_end]]
            elif split == 'val':
                self.data_frame = self.data_frame.iloc[indices[train_end:val_end]]
            else:  # test
                self.data_frame = self.data_frame.iloc[indices[val_end:]]

    def __len__(self):
        return len(self.data_frame)
    
    def load_image_and_mask(self, idx):
        """Load image and mask from disk"""
        rgb_file_name = os.path.basename(self.data_frame.iloc[idx]['path'])
        mask_file_name = os.path.basename(self.data_frame.iloc[idx]['path'])

        rgb_full_path = os.path.join(self.rgb_path, rgb_file_name)
        mask_full_path = os.path.join(self.mask_path, mask_file_name)

        image = cv2.imread(rgb_full_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        
        return image, mask