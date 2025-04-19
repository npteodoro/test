from .base import BaseDataset
import torch
import numpy as np
import cv2
import colorsys

class ClassificationDataset(BaseDataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train', 
                 mask_method=None, mask_weight=0.5, class_column='level'):
        super().__init__(csv_file, root_dir, transform, split)
        self.mask_method = mask_method
        self.mask_weight = mask_weight
        self.class_column = class_column
        self.classes = sorted(self.data_frame[self.class_column].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.color_map = self._create_color_map(self.num_classes)

    def _create_color_map(self, num_classes):
        HSVs = [(i / num_classes, 0.7, 0.9) for i in range(num_classes)]
        
        # Convert to RGB and scale to 0-255
        color_map = {}
        for i, cls in enumerate(self.classes):
            rgb = colorsys.hsv_to_rgb(*HSVs[i])
            # Convert to BGR for OpenCV
            color_map[cls] = [int(255 * c) for c in reversed(rgb)]
            
        return color_map

    def __getitem__(self, idx):
        image, mask = self.load_image_and_mask(idx)
        
        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        # Make sure mask has the same dimensions as image
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
        # Get label
        class_name = self.data_frame.iloc[idx]['level']
        label = self.class_to_idx[class_name]
        
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
        
        if self.mask_method == 'channel':
            image = self.add_mask_as_channel(image, mask)
        elif self.mask_method == 'attention':
            image = self.apply_attention_mask(image, mask)
        elif self.mask_method == 'feature_weighting':
            image = self.apply_feature_weighting(image, mask)
        elif self.mask_method == 'region_focus':
            image = self.apply_region_focus(image, mask)
        elif self.mask_method == 'semantic_aug':
            image = self.apply_semantic_aug(image, mask, class_name)
        
        return image, mask, label   
    
    def add_mask_as_channel(self, image, mask):
        """Simply concatenate mask as an additional channel"""
        if mask.shape[0] != 1:
            mask = mask.unsqueeze(0)
        return torch.cat([image, mask], dim=0)
    
    def apply_attention_mask(self, image, mask):
        """Apply mask as an attention mechanism, highlighting important regions"""
        # Scale mask to [0.5, 1.0] range to avoid completely zeroing out regions
        attention_mask = 0.5 + 0.5 * mask
        
        # Apply attention mask to each channel
        return image * attention_mask
    
    def apply_feature_weighting(self, image, mask):
        """Weight features based on mask importance but preserve original image"""
        weighted_image = (1 - self.mask_weight) * image + self.mask_weight * (image * mask)
        return weighted_image
    
    def apply_region_focus(self, image, mask):
        """Create a dual-stream approach with both masked and unmasked images"""
        masked_region = image * mask
        return torch.cat([image, masked_region], dim=0)
    
    def apply_semantic_aug(self, image, mask, class_name):
        """Apply semantic augmentation based on class label with instance variation"""
        augmented_img = image.copy()
        
        # Get base color for this class
        base_color = self.color_map.get(class_name, [0, 255, 0])
        
        # Option 1: Vary opacity by instance
        # Each instance gets same color but different opacity
        opacity = np.random.uniform(0.2, 0.4)
        
        # Option 2: Add color variation per instance while keeping class identity
        # Create a slightly varied color that still resembles the class color
        variation = 30  # Amount of RGB variation allowed
        instance_color = [
            max(0, min(255, base_color[0] + np.random.randint(-variation, variation))),
            max(0, min(255, base_color[1] + np.random.randint(-variation, variation))),
            max(0, min(255, base_color[2] + np.random.randint(-variation, variation)))
        ]
        
        # Create and apply colored mask
        binary_mask = (mask > 0).astype(np.float32)
        mask_3channel = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        
        # Use either base color or instance color based on your preference
        # colored_mask = mask_3channel * np.array(base_color)  # Option 1: Class-consistent
        colored_mask = mask_3channel * np.array(instance_color)  # Option 2: Instance-varied
        
        mask_region = binary_mask > 0
        augmented_img[mask_region] = (1-opacity) * augmented_img[mask_region] + opacity * colored_mask[mask_region]
        
        return augmented_img.astype(np.uint8)