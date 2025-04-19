from .base import BaseEvaluator
from utils.metrics import dice_coefficient, iou_coefficient
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class CombinedEvaluator(BaseEvaluator):
    """Evaluator for combined segmentation+classification models"""
    
    def evaluate(self, data_loader):
        """Evaluate a combined model"""
        self.model.eval()
        
        # Tracking metrics
        total_dice = 0.0
        total_iou = 0.0
        cls_predictions = []
        cls_targets = []
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for images, masks, labels in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Preprocess masks
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                    
                masks = F.interpolate(masks, size=(128, 128), mode='nearest')
                masks = masks.float()

                # Forward pass
                seg_outputs, cls_outputs, _ = self.model(images)
                seg_outputs = F.interpolate(seg_outputs, size=(128, 128), mode='bilinear', align_corners=False)
                
                # Apply sigmoid and threshold
                outputs_sigmoid = torch.sigmoid(seg_outputs)
                
                # Use the threshold from config if available
                threshold = self.config.get('threshold', 0.5) if self.config else 0.5
                predictions = (outputs_sigmoid > threshold).float()
                
                # Get classification predictions
                _, cls_predicted = torch.max(cls_outputs, 1)
                
                # Calculate metrics
                batch_dice = dice_coefficient(predictions, masks).item()
                batch_iou = iou_coefficient(predictions, masks).item()
                
                # Update metrics
                total_dice += batch_dice
                total_iou += batch_iou
                
                # Store classification results
                cls_predictions.extend(cls_predicted.cpu().numpy())
                cls_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'dice': f"{batch_dice:.4f}",
                    'iou': f"{batch_iou:.4f}"
                })

        # Calculate average metrics
        avg_dice = total_dice / len(data_loader)
        avg_iou = total_iou / len(data_loader)
        
        # Calculate classification metrics
        accuracy = accuracy_score(cls_targets, cls_predictions)
        precision = precision_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        recall = recall_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        f1 = f1_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        
        metrics = {
            'dice': avg_dice,
            'iou': avg_iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Only calculate per-class metrics if requested in config
        if self.config and self.config.get('per_class_metrics', True):
            class_precision = precision_score(cls_targets, cls_predictions, average=None, zero_division=0)
            class_recall = recall_score(cls_targets, cls_predictions, average=None, zero_division=0)
            class_f1 = f1_score(cls_targets, cls_predictions, average=None, zero_division=0)
            
            metrics.update({
                'class_precision': class_precision.tolist(),
                'class_recall': class_recall.tolist(),
                'class_f1': class_f1.tolist(),
            })
            
        # Only calculate confusion matrix if requested
        if self.config and self.config.get('confusion_matrix', True):
            cm = confusion_matrix(cls_targets, cls_predictions)
            metrics['confusion_matrix'] = cm.tolist()
        
        print(f"Test Metrics - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        print(f"Test Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics