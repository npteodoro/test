from .base import BaseEvaluator
from utils.metrics import dice_coefficient, iou_coefficient
import torch
import torch.nn.functional as F
from tqdm import tqdm

class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation models"""
    
    def evaluate(self, data_loader):
        """Evaluate a segmentation model"""
        self.model.eval()
        total_dice = 0.0
        total_iou = 0.0
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(data_loader, desc="Evaluating")

        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Preprocess masks
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                    
                masks = F.interpolate(masks, size=(128, 128), mode='nearest')
                masks = masks.float()

                # Forward pass
                outputs = self.model(images)
                outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=False)
                
                # Apply sigmoid and threshold
                outputs_sigmoid = torch.sigmoid(outputs)
                
                # Use the threshold from config if available
                threshold = self.config.get('threshold', 0.5) if self.config else 0.5
                predictions = (outputs_sigmoid > threshold).float()
                
                # Calculate metrics
                batch_dice = dice_coefficient(predictions, masks).item()
                batch_iou = iou_coefficient(predictions, masks).item()
                
                # Update metrics
                total_dice += batch_dice
                total_iou += batch_iou
                
                # Update progress bar
                progress_bar.set_postfix({
                    'dice': f"{batch_dice:.4f}",
                    'iou': f"{batch_iou:.4f}"
                })

        # Calculate average metrics
        avg_dice = total_dice / len(data_loader)
        avg_iou = total_iou / len(data_loader)
        
        metrics = {
            'dice': avg_dice,
            'iou': avg_iou
        }
        
        print(f"Test Metrics - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")
        
        return metrics

# Legacy function for backward compatibility
def eval(model, test_loader, device, config=None):
    """Legacy wrapper for the SegmentationEvaluator"""
    evaluator = SegmentationEvaluator(model, device, config)
    return evaluator.evaluate(test_loader)