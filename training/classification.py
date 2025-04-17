from .base import BaseTrainer
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        predictions_list = []
        targets_list = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Handle different dataset output formats
            if len(batch) == 3:  # image, mask, label
                images, _, labels = batch
            else:  # image, label
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            train_loss += loss.item()
            
            # Store predictions and targets for metric calculation
            predictions_list.extend(predicted.cpu().numpy())
            targets_list.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate metrics
        accuracy = accuracy_score(targets_list, predictions_list)
        precision = precision_score(targets_list, predictions_list, average='macro', zero_division=0)
        recall = recall_score(targets_list, predictions_list, average='macro', zero_division=0)
        f1 = f1_score(targets_list, predictions_list, average='macro', zero_division=0)
        
        metrics = {
            'loss': train_loss / len(train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different dataset output formats
                if len(batch) == 3:  # image, mask, label
                    images, _, labels = batch
                else:  # image, label
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                val_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                predictions_list.extend(predicted.cpu().numpy())
                targets_list.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(targets_list, predictions_list)
        precision = precision_score(targets_list, predictions_list, average='macro', zero_division=0)
        recall = recall_score(targets_list, predictions_list, average='macro', zero_division=0)
        f1 = f1_score(targets_list, predictions_list, average='macro', zero_division=0)
        
        metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, config=None):
    """Legacy wrapper function to maintain compatibility"""
    trainer = ClassificationTrainer(model, criterion, optimizer, device, config)
    return trainer.train(train_loader, val_loader, num_epochs)