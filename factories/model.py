from architectures.segmentation import SegmentationModel
from architectures.classification import ClassificationModel
import torch

def create_segmentation_model(config):
    """Create a segmentation model based on configuration"""
    model = SegmentationModel(
        encoder_name=config.get('encoder', 'mobilenet_v3_large'),
        decoder_name=config.get('decoder', 'deeplabv3'),
        num_classes=config.get('num_classes', 1),
        weights=config.get('weights', 'DEFAULT') if config.get('pretrained', True) else None
    )
    
    if 'checkpoint' in config and config['checkpoint']:
        model.load_state_dict(torch.load(config['checkpoint']))
    
    return model

def create_classification_model(config):
    """Create a classification model based on configuration"""
    model = ClassificationModel(
        encoder_name=config.get('encoder', 'mobilenet_v3_large'),
        num_classes=config.get('num_classes_cls', 4),
        pretrained=config.get('pretrained', True),
        use_mask_channel=config.get('use_mask_as_channel', False)
    )
    
    if 'checkpoint' in config and config['checkpoint']:
        model.load_state_dict(torch.load(config['checkpoint']))
    
    return model

def create_model(task_type, config):
    """Factory function to create a model based on task type and configuration"""
    if task_type == 'segmentation':
        return create_segmentation_model(config)
    elif task_type == 'classification':
        return create_classification_model(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}. Must be 'segmentation' or 'classification'")