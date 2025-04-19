from .segmentation import SegmentationInference
from .classification import ClassificationInference
from .combined import CombinedInference

def create_inferencer(task_type, model_path, config=None, device=None):
    """Factory function to create an inference object based on task type"""
    if task_type == 'segmentation':
        return SegmentationInference(model_path, device, config)
    elif task_type == 'classification':
        return ClassificationInference(model_path, device, config)
    elif task_type == 'combined':
        return CombinedInference(model_path, device, config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are 'segmentation', 'classification', and 'combined'.")