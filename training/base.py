import torch
import os
import time

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, device, config=None):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}
        self.history = {}
        
        # Learning rate scheduler (optional)
        self.scheduler = self._get_scheduler()
        
        # Setup directories for checkpoints
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _get_scheduler(self):
        """Create learning rate scheduler based on config"""
        scheduler_type = self.config.get('scheduler', None)
        if scheduler_type is None:
            return None
        
        if scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.get('scheduler_mode', 'min'),
                factor=self.config.get('scheduler_factor', 0.1),
                patience=self.config.get('scheduler_patience', 5),
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('scheduler_t_max', 10),
                eta_min=self.config.get('scheduler_eta_min', 0)
            )
        
        return None
    
    def train_epoch(self, data_loader):
        """Base method for training a single epoch"""
        self.model.train()
        epoch_loss = 0
        # Implement in subclasses
        return epoch_loss
    
    def validate(self, data_loader):
        """Base method for validation"""
        self.model.eval()
        val_loss = 0
        # Implement in subclasses
        return val_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """Training loop with validation and early stopping"""
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            # Other metrics will be added by subclasses
        }
        
        best_val_metric = float('inf') if self.config.get('monitor_mode', 'min') == 'min' else float('-inf')
        best_epoch = 0
        early_stop_counter = 0
        early_stop_patience = self.config.get('early_stop_patience', 10)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Update history
            for key in train_metrics:
                if f'train_{key}' not in self.history:
                    self.history[f'train_{key}'] = []
                self.history[f'train_{key}'].append(train_metrics[key])
            
            for key in val_metrics:
                if f'val_{key}' not in self.history:
                    self.history[f'val_{key}'] = []
                self.history[f'val_{key}'].append(val_metrics[key])
            
            # Print progress
            progress = f"Epoch [{epoch+1}/{num_epochs}]"
            for key, value in train_metrics.items():
                progress += f", Train {key}: {value:.4f}"
            for key, value in val_metrics.items():
                progress += f", Val {key}: {value:.4f}"
            print(progress)
            
            # Update learning rate if needed
            if self.scheduler is not None:
                monitor_metric = val_metrics.get(self.config.get('monitor_metric', 'loss'))
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor_metric)
                else:
                    self.scheduler.step()
            
            # Check for improvement
            monitor_metric = val_metrics.get(self.config.get('monitor_metric', 'loss'))
            monitor_mode = self.config.get('monitor_mode', 'min')
            
            improved = (monitor_mode == 'min' and monitor_metric < best_val_metric) or \
                      (monitor_mode == 'max' and monitor_metric > best_val_metric)
            
            if improved:
                best_val_metric = monitor_metric
                best_epoch = epoch
                early_stop_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
            else:
                early_stop_counter += 1
                
                # Save regular checkpoint
                if epoch % self.config.get('checkpoint_interval', 5) == 0:
                    self.save_checkpoint(epoch)
            
            # Early stopping
            if early_stop_counter >= early_stop_patience and self.config.get('early_stopping', False):
                print(f"Early stopping at epoch {epoch+1} as no improvement in {early_stop_patience} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds. Best epoch: {best_epoch+1}")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        return self.model, self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        file_prefix = 'best_model' if is_best else f'model_epoch_{epoch}'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, f"{self.checkpoint_dir}/{file_prefix}.pth")
    
    def load_checkpoint(self, checkpoint_path=None, is_best=False):
        """Load model checkpoint"""
        if checkpoint_path is None and is_best:
            checkpoint_path = f"{self.checkpoint_dir}/best_model.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
            return checkpoint['epoch']
        
        return 0