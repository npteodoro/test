import torch
import os
import time
from tqdm import tqdm

class BaseEvaluator:
    """Base class for all evaluators"""
    
    def __init__(self, model, device, config=None):
        """Initialize the evaluator"""
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        
    def evaluate(self, data_loader):
        """Base method for evaluation. Override in subclasses."""
        self.model.eval()
        # Implement in subclasses
        return {}
    
    def save_results(self, metrics, output_dir='results'):
        """Save evaluation results to a file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create a formatted string of metrics
        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        # Save to file
        model_name = self.config.get('model_name', 'unknown')
        with open(f"{output_dir}/{model_name}_{timestamp}.txt", 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Configuration: {str(self.config)}\n\n")
            f.write(f"Metrics:\n{metrics_str}\n")
            
        return metrics