from evaluation.segmetation import SegmentationEvaluator
from evaluation.classification import ClassificationEvaluator
from evaluation.combined import CombinedEvaluator

def create_evaluator(task_type, model, device, config=None):
    """Create an evaluator based on task type"""
    if task_type == 'segmentation':
        return SegmentationEvaluator(model, device, config)
    elif task_type == 'classification':
        return ClassificationEvaluator(model, device, config)
    elif task_type == 'combined':
        return CombinedEvaluator(model, device, config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are 'segmentation', 'classification', and 'combined'.")