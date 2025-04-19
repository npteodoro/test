import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from PIL import Image

from .base import BaseInference
from factories.model import create_model
from utils.metrics import dice_coefficient, iou_coefficient

class SegmentationInference(BaseInference):
    """Inference module for segmentation models"""
    
    def _load_model(self) -> torch.nn.Module:
        """Load segmentation model from checkpoint"""
        # Load configuration
        config = self.config.get('model', {})
        
        # Create model
        model = create_model('segmentation', config)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Move to device
        model.to(self.device)
        
        return model
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Run segmentation on an image
        
        Args:
            image: Input image
            
        Returns:
            Binary segmentation mask
        """
        # Get original image for later
        if isinstance(image, str):
            original_image = plt.imread(image)
        elif isinstance(image, np.ndarray):
            original_image = image.copy()
        else:
            original_image = np.array(image)
        self.original_size = original_image.shape[:2]
        
        # Preprocess the image
        image_tensor = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            outputs = torch.sigmoid(outputs)
            
            # Apply threshold
            threshold = self.config.get('threshold', 0.5)
            mask = (outputs > threshold).float()
            
            # Convert to numpy
            mask = mask.cpu().numpy()[0, 0]
            
            # Resize to original size if needed
            if mask.shape != self.original_size:
                import cv2
                mask = cv2.resize(mask, (self.original_size[1], self.original_size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            
        return mask
    
    def visualize(self, image: Union[str, np.ndarray, Image.Image], mask: np.ndarray,
                 output_path: Optional[str] = None) -> None:
        """Visualize segmentation mask
        
        Args:
            image: Original image
            mask: Predicted segmentation mask
            output_path: Path to save visualization
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = plt.imread(image)
            
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Create overlay
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        
        # Create mask overlay
        overlay = image.copy()
        overlay = overlay.astype(np.float32) / 255.0 if overlay.max() > 1 else overlay
        
        # Create colored mask (red)
        colored_mask = np.zeros_like(image, dtype=np.float32)
        colored_mask[..., 0] = mask * 1.0  # Red channel
        
        # Apply overlay with transparency
        alpha = 0.5
        overlay = overlay * (1 - alpha * mask[..., None]) + colored_mask * alpha
        
        plt.imshow(np.clip(overlay, 0, 1))
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()