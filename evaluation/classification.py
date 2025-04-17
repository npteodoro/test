from .base import BaseEvaluator
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification models"""
    
    def evaluate(self, data_loader):
        """Evaluate a classification model"""
        self.model.eval()
        predictions_list = []
        targets_list = []
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                # Handle different dataset output formats
                if len(batch) == 3:  # image, mask, label
                    images, _, labels = batch
                else:  # image, label
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions and targets for metric calculation
                predictions_list.extend(predicted.cpu().numpy())
                targets_list.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(targets_list, predictions_list)
        precision = precision_score(targets_list, predictions_list, average='macro', zero_division=0)
        recall = recall_score(targets_list, predictions_list, average='macro', zero_division=0)
        f1 = f1_score(targets_list, predictions_list, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        # Only calculate per-class metrics if requested in config
        if self.config and self.config.get('per_class_metrics', True):
            class_precision = precision_score(targets_list, predictions_list, average=None, zero_division=0)
            class_recall = recall_score(targets_list, predictions_list, average=None, zero_division=0)
            class_f1 = f1_score(targets_list, predictions_list, average=None, zero_division=0)
            
            metrics.update({
                'class_precision': class_precision.tolist(),
                'class_recall': class_recall.tolist(),
                'class_f1': class_f1.tolist(),
            })
            
        # Only calculate confusion matrix if requested
        if self.config and self.config.get('confusion_matrix', True):
            cm = confusion_matrix(targets_list, predictions_list)
            metrics['confusion_matrix'] = cm.tolist()
        
        print(f"Test Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics

# Legacy function for backward compatibility
def eval(model, test_loader, device, config=None):
    """Legacy wrapper for the ClassificationEvaluator"""
    evaluator = ClassificationEvaluator(model, device, config)
    return evaluator.evaluate(test_loader)