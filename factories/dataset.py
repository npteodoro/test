from loaders.segmentation import SegmentationDataset
from loaders.classification import ClassificationDataset
from utils.transforms import get_transforms

def create_segmentation_datasets(csv_file, root_dir, config):
    """Create train, validation, and test segmentation datasets"""
    train_transform = get_transforms({**config, 'augmentation': config.get('augmentation', 'combined')})
    val_transform = get_transforms({**config, 'augmentation': 'base'})
    
    train_dataset = SegmentationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=train_transform,
        split='train'
    )
    
    val_dataset = SegmentationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='val'
    )
    
    test_dataset = SegmentationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='test'
    )
    
    return train_dataset, val_dataset, test_dataset

def create_classification_datasets(csv_file, root_dir, config, mask_method=None):
    """Create train, validation, and test classification datasets"""
    train_transform = get_transforms({**config, 'augmentation': config.get('augmentation', 'combined')})
    val_transform = get_transforms({**config, 'augmentation': 'base'})

    mask_method = config.get('mask_method', mask_method)
    mask_weight = config.get('mask_weight', 0.5)
    class_column = config.get('class_column', 'level')
    
    train_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=train_transform,
        split='train',
        mask_method=mask_method,
        mask_weight=mask_weight,
        class_column=class_column
    )
    
    val_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='val',
        mask_method=mask_method,
        mask_weight=mask_weight,
        class_column=class_column
    )
    
    test_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='test',
        mask_method=mask_method,
        mask_weight=mask_weight,
        class_column=class_column
    )
    
    return train_dataset, val_dataset, test_dataset