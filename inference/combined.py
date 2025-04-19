import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from PIL import Image

from .base import BaseInference
from factories.model import create_model

class CombinedInference(BaseInference):
    """Inference module for combined segmentation+classification models"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None, config: Dict = None):
        super().__init__(model_path, device, config)
        
        # Define class labels
        self.class_labels = self.config.get('class_labels', ['low', 'medium', 'high', 'flood'])
    
    def _load_model(self) -> torch.nn.Module:
        """Load combined model from checkpoint"""
        # Load configuration
        config = self.config.get('model', {})
        
        # Create model
        model = create_model('combined', config)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Move to device
        model.to(self.device)
        
        return model
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Run combined segmentation and classification
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with segmentation mask, class probabilities and predicted class
        """
        # Store original image for visualization
        if isinstance(image, str):
            self.original_image = plt.imread(image)
        elif isinstance(image, np.ndarray):
            self.original_image = image.copy()
        else:
            self.original_image = np.array(image)
        
        self.original_size = self.original_image.shape[:2]
        
        # Preprocess the image
        image_tensor = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            seg_output, cls_output, seg_mask = self.model(image_tensor.to(self.device))
            
            # Process segmentation output
            mask = seg_mask.cpu().numpy()[0, 0]
            
            # Resize mask to original image size
            if mask.shape != self.original_size:
                import cv2
                mask = cv2.resize(mask, (self.original_size[1], self.original_size[0]), 
                                 interpolation=cv2.INTER_LINEAR)
                
            # Apply threshold for binary mask
            threshold = self.config.get('threshold', 0.5)
            binary_mask = (mask > threshold).astype(np.float32)
            
            # Process classification output
            probabilities = torch.softmax(cls_output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return {
            'mask': mask,
            'binary_mask': binary_mask,
            'probabilities': probabilities[0].cpu().numpy(),
            'class_id': predicted_class,
            'class_name': self.class_labels[predicted_class],
            'confidence': float(probabilities[0, predicted_class].item())
        }
    
    def visualize(self, image: Union[str, np.ndarray, Image.Image], 
                 prediction: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """Visualize combined segmentation and classification
        
        Args:
            image: Original image
            prediction: Prediction dictionary from predict()
            output_path: Path to save visualization
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = plt.imread(image)
            
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Segmentation mask
        plt.subplot(2, 2, 2)
        plt.imshow(prediction['mask'], cmap='jet')
        plt.title('Segmentation Mask')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # Overlay
        plt.subplot(2, 2, 3)
        
        # Create colored mask
        overlay = image.copy()
        overlay = overlay.astype(np.float32) / 255.0 if overlay.max() > 1 else overlay
        
        binary_mask = prediction['binary_mask'][..., None]
        colored_mask = np.zeros_like(image, dtype=np.float32)
        colored_mask[..., 0] = binary_mask[..., 0] * 1.0  # Red channel
        
        alpha = 0.5
        overlay = overlay * (1 - alpha * binary_mask) + colored_mask * alpha
        
        plt.imshow(np.clip(overlay, 0, 1))
        plt.title(f"Overlay + Class: {prediction['class_name']}")
        plt.axis('off')
        
        # Classification probabilities
        plt.subplot(2, 2, 4)
        bars = plt.bar(range(len(self.class_labels)), prediction['probabilities'])
        plt.xticks(range(len(self.class_labels)), self.class_labels, rotation=45)
        plt.ylim(0, 1)
        plt.title('Class Probabilities')
        
        # Highlight the predicted class
        bars[prediction['class_id']].set_color('red')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()