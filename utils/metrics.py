import torch

def dice_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred)
    target_sum = torch.sum(target)
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice

def iou_coefficient(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou