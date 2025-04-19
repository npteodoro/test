import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Large_Weights, EfficientNet_B2_Weights, 
    SqueezeNet1_1_Weights, ShuffleNet_V2_X1_0_Weights, 
    MNASNet1_0_Weights
)

class ClassificationModel(nn.Module):
    def __init__(self, encoder_name, num_classes=4, mask_method=None, mask_weight=0.5, 
                 pretrained=True, weights=None, dropout_rate=0.2):
        """
        Enhanced classification model with various mask utilization methods.
        
        Args:
            encoder_name: Name of backbone encoder ('mobilenet_v3_large', etc.)
            num_classes: Number of output classes
            mask_method: How to use mask - None, 'channel', 'attention', 'feature_weighting', 
                         'region_focus', or 'attention_layer'
            mask_weight: Weight to apply when using attention or feature weighting (0.0-1.0)
            pretrained: Whether to use pretrained weights
            weights: Specific weights to use (overrides pretrained if provided)
            dropout_rate: Dropout rate for the final classification layer
        """
        super(ClassificationModel, self).__init__()
        
        self.mask_method = mask_method
        self.mask_weight = mask_weight
        
        # Determine input channels based on mask method
        self.in_channels = 3  # Default RGB
        if mask_method == 'channel':
            self.in_channels = 4  # RGB + Mask
        elif mask_method == 'region_focus':
            self.in_channels = 6  # RGB + RGB*Mask
        
        # Handle pretrained/weights logic
        use_pretrained = pretrained if weights is None else True
        
        # Store encoder creation functions in a dictionary
        encoders = {
            'mobilenet_v3_large': self._create_mobilenet_v3_large_encoder,
            'efficientnet_b2': self._create_efficientnet_b2_encoder,
            'squeezenet1_1': self._create_squeezenet1_1_encoder,
            'shufflenet_v2_x1_0': self._create_shufflenet_v2_x1_0_encoder,
            'mnasnet1_0': self._create_mnasnet1_0_encoder,
        }
        
        # Create encoder and feature extractor
        self.encoder = encoders[encoder_name](use_pretrained)
        self.feature_extractor = self._create_feature_extractor(encoder_name)
        
        # Create classifier with dropout
        in_features = self._get_classifier_in_features(encoder_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        # Add attention mechanism if needed
        if mask_method == 'attention_layer':
            # Get first layer information based on architecture
            first_layer_info = {
                'mobilenet_v3_large': ('features', 0, 0, 16),  # (module, layer1, layer2, channels)
                'efficientnet_b2': ('features', 0, 0, 32),
                'squeezenet1_1': ('features', 0, None, 64),
                'shufflenet_v2_x1_0': ('conv1', 0, None, 24),
                'mnasnet1_0': ('layers', 0, None, 32),
            }
            
            module, layer1, layer2, channels = first_layer_info[encoder_name]
            
            # Create attention modules
            self.attention_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.attention_bn = nn.BatchNorm2d(channels)
            self.attention_relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        """
        Forward pass with optional mask for attention mechanisms
        
        Args:
            x: Input tensor (images)
            mask: Optional mask tensor for attention_layer method
        """
        # Special handling for attention_layer method
        if self.mask_method == 'attention_layer' and mask is not None:
            # This implementation would need to be customized based on network architecture
            # Here's a general approach that you'd need to adapt:
            
            # Get the first few layers until the point we want to apply attention
            # This is architecture-specific and would need custom implementation for each
            # For this example, let's assume we've captured the point to apply attention
            
            features = self.encoder.features[0](x)  # First block output
            
            # Resize mask to match feature dimensions
            if mask.shape[2:] != features.shape[2:]:
                mask_resized = nn.functional.interpolate(
                    mask, size=features.shape[2:], mode='nearest')
            else:
                mask_resized = mask
                
            # Apply attention
            attention = self.attention_conv(features * mask_resized)
            attention = self.attention_bn(attention)
            attention = self.attention_relu(attention)
            
            # Apply attention to features
            features = features * attention
            
            # Continue with the rest of the network
            # This would need to be customized based on the specific architecture
            # For now, we'll use our feature extractor which expects the full encoder
            
            x = self.feature_extractor(self.encoder, x)
            
        else:
            # Standard forward pass
            x = self.feature_extractor(self.encoder, x)
            
        return self.classifier(x)
    
    def _create_mobilenet_v3_large_encoder(self, pretrained):
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        if self.in_channels != 3:
            original_conv = model.features[0][0]
            model.features[0][0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_efficientnet_b2_encoder(self, pretrained):
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        
        if self.in_channels != 3:
            original_conv = model.features[0][0]
            model.features[0][0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_squeezenet1_1_encoder(self, pretrained):
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        
        if self.in_channels != 3:
            original_conv = model.features[0]
            model.features[0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_shufflenet_v2_x1_0_encoder(self, pretrained):
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        
        if self.in_channels != 3:
            original_conv = model.conv1[0]
            model.conv1[0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_mnasnet1_0_encoder(self, pretrained):
        weights = MNASNet1_0_Weights.DEFAULT if pretrained else None
        model = models.mnasnet1_0(weights=weights)
        
        if self.in_channels != 3:
            original_conv = model.layers[0]
            model.layers[0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_feature_extractor(self, encoder_name):
        """Create a feature extraction function for the given encoder"""
        def extract_mobilenet_v3_large(encoder, x):
            x = encoder.features(x)
            x = encoder.avgpool(x)
            x = torch.flatten(x, 1)
            return x
            
        def extract_efficientnet_b2(encoder, x):
            x = encoder.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
            
        def extract_squeezenet1_1(encoder, x):
            x = encoder.features(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
            
        def extract_shufflenet_v2_x1_0(encoder, x):
            x = encoder.conv1(x)
            x = encoder.maxpool(x)
            x = encoder.stage2(x)
            x = encoder.stage3(x)
            x = encoder.stage4(x)
            x = encoder.conv5(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
            
        def extract_mnasnet1_0(encoder, x):
            x = encoder.layers(x)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            return x
            
        extractors = {
            'mobilenet_v3_large': extract_mobilenet_v3_large,
            'efficientnet_b2': extract_efficientnet_b2,
            'squeezenet1_1': extract_squeezenet1_1,
            'shufflenet_v2_x1_0': extract_shufflenet_v2_x1_0,
            'mnasnet1_0': extract_mnasnet1_0,
        }
        
        return extractors[encoder_name]
    
    def _get_classifier_in_features(self, encoder_name):
        """Return the number of input features for the classifier"""
        features = {
            'mobilenet_v3_large': 960,
            'efficientnet_b2': 1408,
            'shufflenet_v2_x1_0': 1024,
            'mnasnet1_0': 1280,
            'squeezenet1_1': 512,
        }
        
        return features[encoder_name]
    
    def _modify_first_conv_layer(self, original_conv):
        """Modify the first convolutional layer to accept additional channels"""
        new_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None
        )

        with torch.no_grad():
            # Clone existing weights for first 3 channels
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
            
            # Initialize additional channels - use mean of RGB channels
            if self.in_channels > 3:
                # For mask channel or other additional channels
                for c in range(3, self.in_channels):
                    new_conv.weight[:, c:c+1, :, :] = original_conv.weight[:, :3, :, :].mean(dim=1, keepdim=True)

            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.clone())

        return new_conv