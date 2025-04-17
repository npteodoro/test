from .base import BaseTrainer
import torch
import torch.nn.functional as F
from utils.metrics import dice_coefficient, iou_coefficient
from tqdm import tqdm

class SegmentationTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Preprocess masks
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            masks = F.interpolate(masks, size=(128, 128), mode='nearest')
            masks = masks.float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=False)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            predictions = (outputs_sigmoid > 0.5).float()
            
            batch_dice = dice_coefficient(predictions, masks).item()
            batch_iou = iou_coefficient(predictions, masks).item()
            
            # Update metrics
            train_loss += loss.item()
            train_dice += batch_dice
            train_iou += batch_iou
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_dice:.4f}",
                'iou': f"{batch_iou:.4f}"
            })
        
        # Calculate average metrics
        metrics = {
            'loss': train_loss / len(train_loader),
            'dice': train_dice / len(train_loader),
            'iou': train_iou / len(train_loader)
        }
        
        return metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
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
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                outputs_sigmoid = torch.sigmoid(outputs)
                predictions = (outputs_sigmoid > 0.5).float()
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice_coefficient(predictions, masks).item()
                val_iou += iou_coefficient(predictions, masks).item()
        
        # Calculate average metrics
        metrics = {
            'loss': val_loss / len(val_loader),
            'dice': val_dice / len(val_loader),
            'iou': val_iou / len(val_loader)
        }
        
        return metrics

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, config=None):
    """Legacy wrapper function to maintain compatibility"""
    trainer = SegmentationTrainer(model, criterion, optimizer, device, config)
    return trainer.train(train_loader, val_loader, num_epochs)