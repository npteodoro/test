from .base import BaseDataset
import torch
import numpy as np
import cv2

class ClassificationDataset(BaseDataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train', 
                 use_mask_as_channel=False, use_mask_for_aug=False):
        super().__init__(csv_file, root_dir, transform, split)
        self.use_mask_as_channel = use_mask_as_channel
        self.use_mask_for_aug = use_mask_for_aug
        self.classes = ['low', 'medium', 'high', 'flood']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __getitem__(self, idx):
        image, mask = self.load_image_and_mask(idx)
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Make sure mask has the same dimensions as image
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Apply semantic augmentation if needed
        if self.use_mask_for_aug:
            image = self.apply_semantic_aug(image, mask, idx)
        
        # Apply transforms
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        
        if not isinstance(mask, torch.Tensor):
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            mask = torch.from_numpy(mask).unsqueeze(0) / 255.0
        
        # Add mask as channel if needed
        if self.use_mask_as_channel:
            if mask.shape[0] != 1:
                mask = mask.unsqueeze(0)
            image = torch.cat([image, mask], dim=0)
        
        # Get label
        class_name = self.data_frame.iloc[idx]['level']
        label = self.class_to_idx[class_name]
        
        return image, mask, label   
    
    def apply_semantic_aug(self, image, mask, idx):
        augmented_img = image.copy()
        class_name = self.data_frame.iloc[idx]['level']
        
        if class_name == 'low':
            overlay_color = [0, 0, 255]
            opacity = np.random.uniform(0.2, 0.4)
        elif class_name == 'medium':
            overlay_color = [0, 255, 0]
            opacity = np.random.uniform(0.2, 0.4)
        elif class_name == 'high':
            overlay_color = [255, 0, 0]
            opacity = np.random.uniform(0.2, 0.4)
        elif class_name == 'flood':
            overlay_color = [255, 255, 0]
            opacity = np.random.uniform(0.2, 0.4)
        else:
            overlay_color = np.random.randint(0, 255, 3).tolist()
            opacity = np.random.uniform(0.2, 0.4)
            
        binary_mask = (mask > 0).astype(np.float32)
        mask_3channel = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        colored_mask = mask_3channel * np.array(overlay_color)
            
        mask_region = binary_mask > 0
        augmented_img[mask_region] = (1-opacity) * augmented_img[mask_region] + opacity * colored_mask[mask_region]
        
        return augmented_img.astype(np.uint8)