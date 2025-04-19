import torch
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
import matplotlib.pyplot as plt

class BaseInference:
    """Base class for all inference implementations"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None, config: Dict = None):
        """Initialize the inference module
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on (defaults to CUDA if available)
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint - override in subclasses"""
        raise NotImplementedError("Subclasses must implement _load_model")
        
    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess an image for inference
        
        Args:
            image: Image as file path, numpy array, or PIL Image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Basic preprocessing - resize and normalize
        image = cv2.resize(image, (self.config.get('image_size', 128), self.config.get('image_size', 128)))
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Any:
        """Run inference on an image
        
        Args:
            image: Input image (path, numpy array, or PIL image)
            
        Returns:
            Prediction result (format depends on model type)
        """
        # Preprocess the image
        image_tensor = self.preprocess(image)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            
        return output
    
    def visualize(self, image: Union[str, np.ndarray, Image.Image], prediction: Any, 
                 output_path: Optional[str] = None) -> None:
        """Visualize predictions - override in subclasses
        
        Args:
            image: Original image
            prediction: Model prediction
            output_path: Path to save visualization (if None, shows the plot)
        """
        raise NotImplementedError("Subclasses must implement visualize")
        
    def batch_predict(self, image_dir: str, output_dir: Optional[str] = None) -> List[Any]:
        """Run inference on all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results (if None, no saving)
            
        Returns:
            List of prediction results
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # Run inference
            prediction = self.predict(image_path)
            results.append(prediction)
            
            # Save visualization if output directory is provided
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_result.png")
                self.visualize(image_path, prediction, output_path)
                
        return results