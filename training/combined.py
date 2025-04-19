from .base import BaseTrainer
import torch
import torch.nn.functional as F
from utils.metrics import dice_coefficient, iou_coefficient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np

class CombinedTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, device, config=None):
        super().__init__(model, criterion, optimizer, device, config)
        # Create separate criterions for segmentation and classification
        self.seg_criterion = torch.nn.BCEWithLogitsLoss()
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        
        # Loss weighting
        self.seg_weight = config.get('seg_loss_weight', 0.5)
        self.cls_weight = config.get('cls_loss_weight', 0.5)
        
        # Print weights for debugging
        print(f"Segmentation loss weight: {self.seg_weight}")
        print(f"Classification loss weight: {self.cls_weight}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        # Tracking metrics
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_total_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        cls_predictions = []
        cls_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, masks, labels in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Preprocess masks - ensure they are binary and properly shaped
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            masks = F.interpolate(masks, size=(128, 128), mode='nearest')
            # Ensure masks are binary 0.0 or 1.0
            masks = (masks > 0.5).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            seg_outputs, cls_outputs, _ = self.model(images)
            
            # Resize segmentation outputs
            seg_outputs = F.interpolate(seg_outputs, size=(128, 128), 
                                       mode='bilinear', align_corners=False)
            
            # Clamp outputs to avoid extreme values
            seg_outputs = torch.clamp(seg_outputs, min=-50.0, max=50.0)
            
            # Calculate losses with proper type checking
            seg_loss = self.seg_criterion(seg_outputs, masks)
            cls_loss = self.cls_criterion(cls_outputs, labels)
            
            # Safety check - if loss is negative (which shouldn't happen), use abs value
            if seg_loss < 0:
                print(f"Warning: Negative segmentation loss detected: {seg_loss.item()}")
                seg_loss = torch.abs(seg_loss)
            
            # Combined loss
            loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            seg_predictions = (torch.sigmoid(seg_outputs) > 0.5).float()
            batch_dice = dice_coefficient(seg_predictions, masks).item()
            batch_iou = iou_coefficient(seg_predictions, masks).item()
            
            # Get classification predictions
            _, predicted = torch.max(cls_outputs, 1)
            
            # Update metrics
            train_seg_loss += seg_loss.item()
            train_cls_loss += cls_loss.item()
            train_total_loss += loss.item()
            train_dice += batch_dice
            train_iou += batch_iou
            
            cls_predictions.extend(predicted.cpu().numpy())
            cls_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'seg_loss': f"{seg_loss.item():.4f}",
                'cls_loss': f"{cls_loss.item():.4f}",
                'dice': f"{batch_dice:.4f}"
            })
        
        # Calculate classification metrics
        accuracy = accuracy_score(cls_targets, cls_predictions)
        precision = precision_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        recall = recall_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        f1 = f1_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        
        # Calculate average metrics
        metrics = {
            'loss': train_total_loss / len(train_loader),
            'seg_loss': train_seg_loss / len(train_loader),
            'cls_loss': train_cls_loss / len(train_loader),
            'dice': train_dice / len(train_loader),
            'iou': train_iou / len(train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        # Tracking metrics
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        val_total_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        cls_predictions = []
        cls_targets = []
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Preprocess masks - ensure they are binary and properly shaped
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                
                masks = F.interpolate(masks, size=(128, 128), mode='nearest')
                # Ensure masks are binary 0.0 or 1.0
                masks = (masks > 0.5).float()
                
                # Forward pass
                seg_outputs, cls_outputs, _ = self.model(images)
                
                # Resize segmentation outputs
                seg_outputs = F.interpolate(seg_outputs, size=(128, 128), 
                                           mode='bilinear', align_corners=False)
                
                # Clamp outputs to avoid extreme values
                seg_outputs = torch.clamp(seg_outputs, min=-50.0, max=50.0)
                
                # Calculate losses with proper type checking
                seg_loss = self.seg_criterion(seg_outputs, masks)
                cls_loss = self.cls_criterion(cls_outputs, labels)
                
                # Safety check - if loss is negative (which shouldn't happen), use abs value
                if seg_loss < 0:
                    print(f"Warning: Negative validation seg loss: {seg_loss.item()}")
                    seg_loss = torch.abs(seg_loss)
                
                # Combined loss
                loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
                
                # Calculate metrics
                seg_predictions = (torch.sigmoid(seg_outputs) > 0.5).float()
                batch_dice = dice_coefficient(seg_predictions, masks).item()
                batch_iou = iou_coefficient(seg_predictions, masks).item()
                
                # Get classification predictions
                _, predicted = torch.max(cls_outputs, 1)
                
                # Update metrics
                val_seg_loss += seg_loss.item()
                val_cls_loss += cls_loss.item()
                val_total_loss += loss.item()
                val_dice += batch_dice
                val_iou += batch_iou
                
                cls_predictions.extend(predicted.cpu().numpy())
                cls_targets.extend(labels.cpu().numpy())
        
        # Calculate classification metrics
        accuracy = accuracy_score(cls_targets, cls_predictions)
        precision = precision_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        recall = recall_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        f1 = f1_score(cls_targets, cls_predictions, average='macro', zero_division=0)
        
        # Calculate average metrics
        metrics = {
            'loss': val_total_loss / len(val_loader),
            'seg_loss': val_seg_loss / len(val_loader),
            'cls_loss': val_cls_loss / len(val_loader),
            'dice': val_dice / len(val_loader),
            'iou': val_iou / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics