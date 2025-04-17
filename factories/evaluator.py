from evaluation.segmetation import SegmentationEvaluator
from evaluation.classification import ClassificationEvaluator

def create_evaluator(task_type, model, device, config=None):
    """Create an evaluator based on task type"""
    if task_type == 'segmentation':
        return SegmentationEvaluator(model, device, config)
    elif task_type == 'classification':
        return ClassificationEvaluator(model, device, config)
    else:
        raise ValueError(f"Unknown task type: {task_type}. Must be 'segmentation' or 'classification'")