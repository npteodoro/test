from training.segmentation import SegmentationTrainer
from training.classification import ClassificationTrainer
from training.combined import CombinedTrainer

def create_trainer(task_type, model, criterion, optimizer, device, config=None):
    """Create a trainer based on task type"""
    if task_type == 'segmentation':
        return SegmentationTrainer(model, criterion, optimizer, device, config)
    elif task_type == 'classification':
        return ClassificationTrainer(model, criterion, optimizer, device, config)
    elif task_type == 'combined':
        return CombinedTrainer(model, criterion, optimizer, device, config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are 'segmentation', 'classification', and 'combined'.")