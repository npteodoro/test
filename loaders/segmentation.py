from .base import BaseDataset
import torch.nn.functional as F
import torch
import numpy as np
import cv2

class SegmentationDataset(BaseDataset):
    def __getitem__(self, idx):
        image_np, mask_np = self.load_image_and_mask(idx)
        
        # Process mask
        mask_np = (mask_np > 127).astype(np.uint8)
        
        # Handle different sizes
        if image_np.shape[:2] != mask_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        image, mask = self.transform(image_np, mask_np)

        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        # Final processing
        mask = mask.unsqueeze(0)
        
        return image, mask