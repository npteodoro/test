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

def create_classification_datasets(csv_file, root_dir, config, use_mask=False):
    """Create train, validation, and test classification datasets"""
    train_transform = get_transforms({**config, 'augmentation': config.get('augmentation', 'combined')})
    val_transform = get_transforms({**config, 'augmentation': 'base'})
    
    train_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=train_transform,
        split='train',
        use_mask_as_channel=use_mask,
        use_mask_for_aug=config.get('use_mask_for_aug', False)
    )
    
    val_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='val',
        use_mask_as_channel=use_mask
    )
    
    test_dataset = ClassificationDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=val_transform,
        split='test',
        use_mask_as_channel=use_mask
    )
    
    return train_dataset, val_dataset, test_dataset