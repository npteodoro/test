from torch.utils.data import DataLoader
import torch

def create_data_loader(dataset, config, shuffle=False, batch_size=None):
    """Create a DataLoader with the given configuration"""
    return DataLoader(
        dataset,
        batch_size=batch_size or config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False
    )