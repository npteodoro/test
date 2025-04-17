import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_base_transforms(image_np, mask_np):
    transform = A.Compose([
        A.Resize(height=128, width=128),  # Fix: use consistent size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image_np, mask=mask_np)
    return transformed['image'], transformed['mask']

def get_geometric_transforms(image_np, mask_np):
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),  # equivalent to scale_limit=0.1
            translate_percent=(-0.0625, 0.0625),  # equivalent to shift_limit=0.0625
            rotate=(-45, 45),  # equivalent to rotate_limit=45
            p=0.5
        ),
        A.Resize(height=128, width=128),  # Fix: use consistent size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image_np, mask=mask_np)
    return transformed['image'], transformed['mask']

def get_semantic_transforms(image_np, mask_np):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.2),
        A.CLAHE(p=0.5),
        A.Resize(height=128, width=128),  # Fix: use consistent size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image_np, mask=mask_np)

    return transformed['image'], transformed['mask']

def get_combined_transforms(image_np, mask_np):
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.2),
        A.CLAHE(p=0.3),
        # Change this to ensure the image matches your model output size
        A.Resize(height=128, width=128),  # Fix: remove always_apply and specify height/width
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image_np, mask=mask_np)
    return transformed['image'], transformed['mask']


def get_transforms(config):
    if config['augmentation'] == 'geometric':
        return get_geometric_transforms
    elif config['augmentation'] == 'semantic':
        return get_semantic_transforms
    elif config['augmentation'] == 'combined':
        return get_combined_transforms
    else:
        return get_base_transforms