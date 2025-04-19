from architectures.segmentation import SegmentationModel
from architectures.classification import ClassificationModel
from architectures.combined import CombinedModel
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
        mask_method=config.get('mask_method', None),
        mask_weight=config.get('mask_weight', 0.5),
        dropout_rate=config.get('dropout_rate', 0.2)
    )
    
    if 'checkpoint' in config and config['checkpoint']:
        model.load_state_dict(torch.load(config['checkpoint']))
    
    return model

def create_combined_model(config):
    """Create a combined model based on configuration"""
    model = CombinedModel(
        encoder_name=config.get('encoder', 'mobilenet_v3_large'),
        decoder_name=config.get('decoder', 'deeplabv3'),
        num_classes_seg=config.get('num_classes_seg', 1),
        num_classes_cls=config.get('num_classes_cls', 4),
        shared_encoder=config.get('shared_encoder', True),
        pretrained=config.get('pretrained', True),
        mask_method=config.get('mask_method', 'channel'),
        mask_weight=config.get('mask_weight', 0.5),
        dropout_rate=config.get('dropout_rate', 0.2)
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
    elif task_type == 'combined':
        return create_combined_model(config)
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types are 'segmentation', 'classification', and 'combined'.")