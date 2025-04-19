import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from PIL import Image

from .base import BaseInference
from factories.model import create_model

class ClassificationInference(BaseInference):
    """Inference module for classification models"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None, config: Dict = None):
        super().__init__(model_path, device, config)
        
        # Define class labels
        self.class_labels = self.config.get('class_labels', ['low', 'medium', 'high', 'flood'])
    
    def _load_model(self) -> torch.nn.Module:
        """Load classification model from checkpoint"""
        # Load configuration
        config = self.config.get('model', {})
        
        # Create model
        model = create_model('classification', config)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Move to device
        model.to(self.device)
        
        return model
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], 
               mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run classification on an image
        
        Args:
            image: Input image
            mask: Optional segmentation mask (for mask-based methods)
            
        Returns:
            Dictionary with class probabilities and predicted class
        """
        # Store original image for visualization
        if isinstance(image, str):
            self.original_image = plt.imread(image)
        elif isinstance(image, np.ndarray):
            self.original_image = image.copy()
        else:
            self.original_image = np.array(image)
        
        # Preprocess the image
        image_tensor = self.preprocess(image)
        
        # Add mask as channel if using mask_method == 'channel'
        if mask is not None and self.config.get('mask_method') == 'channel':
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            if mask_tensor.shape[-2:] != image_tensor.shape[-2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor, size=image_tensor.shape[-2:], mode='nearest')
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=1)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return {
            'probabilities': probabilities[0].cpu().numpy(),
            'class_id': predicted_class,
            'class_name': self.class_labels[predicted_class],
            'confidence': float(probabilities[0, predicted_class].item())
        }
    
    def visualize(self, image: Union[str, np.ndarray, Image.Image], 
                 prediction: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """Visualize classification result
        
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
            
        plt.figure(figsize=(10, 6))
        
        # Display image with prediction
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Predicted: {prediction['class_name']}\nConfidence: {prediction['confidence']:.2%}")
        plt.axis('off')
        
        # Display probability distribution
        plt.subplot(1, 2, 2)
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