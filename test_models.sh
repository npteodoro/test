#!/bin/bash
# test_best_models.sh

# Create results directory
RESULTS_DIR="results/best_models_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Number of epochs
EPOCHS=25

echo "Running recommended model configurations with $EPOCHS epochs each"

# Segmentation models
echo "==== TESTING SEGMENTATION MODELS ===="

# 1. EfficientNet + DeepLabV3 + Combined
echo "Running EfficientNet + DeepLabV3 + Combined"
python main.py --task segmentation \
  --set model.encoder efficientnet_b2 \
  --set model.decoder deeplabv3 \
  --set data.augmentation combined \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/seg_efficientnet_deeplabv3_combined.log

# 2. MobileNet + UNet + Combined
echo "Running MobileNet + UNet + Combined"
python main.py --task segmentation \
  --set model.encoder mobilenet_v3_large \
  --set model.decoder unet \
  --set data.augmentation combined \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/seg_mobilenet_unet_combined.log

# 3. MobileNet + FPN + Geometric
echo "Running MobileNet + FPN + Geometric"
python main.py --task segmentation \
  --set model.encoder mobilenet_v3_large \
  --set model.decoder fpn \
  --set data.augmentation geometric \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/seg_mobilenet_fpn_geometric.log

# Classification models
echo "==== TESTING CLASSIFICATION MODELS ===="

# 1. EfficientNet + Mask + Combined
echo "Running EfficientNet + Mask + Combined"
python main.py --task classification \
  --set model.encoder efficientnet_b2 \
  --set model.use_mask_as_channel true \
  --set data.use_mask_as_channel true \
  --set data.augmentation combined \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/cls_efficientnet_mask_combined.log

# 2. MobileNet + Mask + Combined
echo "Running MobileNet + Mask + Combined"
python main.py --task classification \
  --set model.encoder mobilenet_v3_large \
  --set model.use_mask_as_channel true \
  --set data.use_mask_as_channel true \
  --set data.augmentation combined \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/cls_mobilenet_mask_combined.log

# 3. EfficientNet + No Mask + Combined
echo "Running EfficientNet + No Mask + Combined"
python main.py --task classification \
  --set model.encoder efficientnet_b2 \
  --set model.use_mask_as_channel false \
  --set data.use_mask_as_channel false \
  --set data.augmentation combined \
  --set training.num_epochs $EPOCHS \
  --set training.save_checkpoints false > $RESULTS_DIR/cls_efficientnet_nomask_combined.log

echo "All tests completed. Results saved to $RESULTS_DIR"