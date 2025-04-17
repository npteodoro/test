import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    MobileNet_V3_Large_Weights, EfficientNet_B2_Weights, 
    SqueezeNet1_1_Weights, ShuffleNet_V2_X1_0_Weights, 
    MNASNet1_0_Weights
)

class ClassificationModel(nn.Module):
    def __init__(self, encoder_name, num_classes=4, use_mask_channel=False, pretrained=True, weights=None):
        super(ClassificationModel, self).__init__()
        
        self.use_mask_channel = use_mask_channel
        self.in_channels = 4 if use_mask_channel else 3
        
        # Handle pretrained/weights logic like SegmentationModel
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
        
        # Create classifier
        self.classifier = nn.Linear(self._get_classifier_in_features(encoder_name), num_classes)

    def forward(self, x):
        features = self.feature_extractor(self.encoder, x)
        return self.classifier(features)
    
    def _create_mobilenet_v3_large_encoder(self, pretrained):
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        if self.use_mask_channel:
            original_conv = model.features[0][0]
            model.features[0][0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_efficientnet_b2_encoder(self, pretrained):
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        
        if self.use_mask_channel:
            original_conv = model.features[0][0]
            model.features[0][0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_squeezenet1_1_encoder(self, pretrained):
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        model = models.squeezenet1_1(weights=weights)
        
        if self.use_mask_channel:
            original_conv = model.features[0]
            model.features[0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_shufflenet_v2_x1_0_encoder(self, pretrained):
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        
        if self.use_mask_channel:
            original_conv = model.conv1[0]
            model.conv1[0] = self._modify_first_conv_layer(original_conv)
            
        return model
    
    def _create_mnasnet1_0_encoder(self, pretrained):
        weights = MNASNet1_0_Weights.DEFAULT if pretrained else None
        model = models.mnasnet1_0(weights=weights)
        
        if self.use_mask_channel:
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
            'mobilenet_v3_large': 960,   # Updated
            'efficientnet_b2': 1408,     # Updated
            'shufflenet_v2_x1_0': 1024,  # Updated
            'mnasnet1_0': 1280,          # Updated
            'squeezenet1_1': 512,        # Same
        }
        
        return features[encoder_name]
    
    def _modify_first_conv_layer(self, original_conv):
        """Modify the first convolutional layer to accept an additional channel"""
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
            # Clone existing weights
            new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
            
            # Initialize the new channel - use mean of RGB channels
            new_conv.weight[:, 3:, :, :] = original_conv.weight[:, :3, :, :].mean(dim=1, keepdim=True)

            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.clone())

        return new_conv

