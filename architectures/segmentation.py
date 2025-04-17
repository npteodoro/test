import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights, EfficientNet_B2_Weights, 
    SqueezeNet1_1_Weights, ShuffleNet_V2_X1_0_Weights, 
    MNASNet1_0_Weights
)

class SegmentationModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, num_classes, pretrained=True, weights=None):
        super(SegmentationModel, self).__init__()
        
        use_pretrained = pretrained if weights is None else True
        
        encoders = {
            'mobilenet_v3_large': self._create_mobilenet_v3_large_encoder(use_pretrained),
            'efficientnet_b2': self._create_efficientnet_b2_encoder(use_pretrained),
            'squeezenet1_1': self._create_squeezenet1_1_encoder(use_pretrained),
            'shufflenet_v2_x1_0': self._create_shufflenet_v2_x1_0_encoder(use_pretrained),
            'mnasnet1_0': self._create_mnasnet1_0_encoder(use_pretrained),
        }

        self.encoder = encoders.get(encoder_name)
        self.encoder_channels = self._get_encoder_channels(encoder_name)

        decoders = {
            'unet': self._create_unet_decoder,
            'fpn': self._create_fpn_decoder,
            'deeplabv3': self._create_deeplabv3_decoder,
        }

        self.decoder = decoders[decoder_name](num_classes)

    def forward(self, x):
        features = []

        if isinstance(self.encoder, models.mobilenet.MobileNetV3):
            x = self.encoder.features[0](x)
            features.append(x)
            
            stages = [3, 6, 12, 16]
                
            for i in range(1, len(self.encoder.features)):
                x = self.encoder.features[i](x)
                if i in stages:
                    features.append(x)
                    
        elif isinstance(self.encoder, models.efficientnet.EfficientNet):
            x = self.encoder.features[0](x)
            features.append(x)
            
            blocks = [2, 3, 5, 8]
            for i in range(1, 9):
                x = self.encoder.features[i](x)
                if i in blocks:
                    features.append(x)
                    
        elif isinstance(self.encoder, models.squeezenet.SqueezeNet):
            x = self.encoder.features[0](x)
            x = self.encoder.features[1](x)
            features.append(x)
            
            stages = [3, 7, 12]
                
            for i in range(2, len(self.encoder.features)):
                x = self.encoder.features[i](x)
                if i in stages:
                    features.append(x)
            
            features.append(x)
            
        elif isinstance(self.encoder, models.shufflenetv2.ShuffleNetV2):
            x = self.encoder.conv1(x)
            x = self.encoder.maxpool(x)
            features.append(x)
            
            x = self.encoder.stage2(x)
            features.append(x)
            
            x = self.encoder.stage3(x)
            features.append(x)
            
            x = self.encoder.stage4(x)
            features.append(x)
            
            x = self.encoder.conv5(x)
            features.append(x)
            
        elif isinstance(self.encoder, models.mnasnet.MNASNet):
            x = self.encoder.layers[0](x)
            features.append(x)
            
            stages = [3, 8, 12, 16]
            for i in range(1, len(self.encoder.layers)):
                x = self.encoder.layers[i](x)
                if i in stages:
                    features.append(x)
        
        out = self.decoder(features)
        return out

    
    def _get_encoder_channels(self, encoder_name):
        channels = {
            'mobilenet_v3_large': [16, 24, 40, 112, 960],
            'efficientnet_b2': [32, 24, 48, 112, 1408],
            'squeezenet1_1': [64, 128, 256, 512, 512],
            'shufflenet_v2_x1_0': [24, 116, 232, 464, 1024],
            'mnasnet1_0': [16, 32, 40, 96, 1280],
        }

        return channels[encoder_name]
    
    def _create_mobilenet_v3_large_encoder(self, pretrained):
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        return models.mobilenet_v3_large(weights=weights)
    
    def _create_efficientnet_b2_encoder(self, pretrained):
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b2(weights=weights)
    
    def _create_squeezenet1_1_encoder(self, pretrained):
        weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        return models.squeezenet1_1(weights=weights)
    
    def _create_shufflenet_v2_x1_0_encoder(self, pretrained):
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        return models.shufflenet_v2_x1_0(weights=weights)
    
    def _create_mnasnet1_0_encoder(self, pretrained):
        weights = MNASNet1_0_Weights.DEFAULT if pretrained else None
        return models.mnasnet1_0(weights=weights)
    
    def _create_unet_decoder(self, num_classes):
        return UNetDecoder(self.encoder_channels, num_classes)
    
    def _create_fpn_decoder(self, num_classes):
        return FPNDecoder(self.encoder_channels, num_classes)
    
    def _create_deeplabv3_decoder(self, num_classes):
        return DeepLabV3Decoder(self.encoder_channels, num_classes)
    
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super(UNetDecoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(encoder_channels) - 1, 0, -1):
            in_channels = encoder_channels[i] + encoder_channels[i - 1]
            out_channels = encoder_channels[i - 1]
            self.decoder_blocks.append(self._decoder_block(in_channels, out_channels))
        
        self.final_conv = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        
        x = features[-1]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            encoder_feature = features[-(i + 2)]
            
            if x.size()[2:] != encoder_feature.size()[2:]:
                x = F.interpolate(x, size=encoder_feature.size()[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, encoder_feature], dim=1)
            
            x = decoder_block(x)
        
        x = self.final_conv(x)
        
        return x
    
class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super(FPNDecoder, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(encoder_channels[1], 256, kernel_size=1),
            nn.Conv2d(encoder_channels[2], 256, kernel_size=1),
            nn.Conv2d(encoder_channels[3], 256, kernel_size=1),
            nn.Conv2d(encoder_channels[4], 256, kernel_size=1)
        ])
        
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        ])
        
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        
    def forward(self, features):
        laterals = []
        
        for i, conv in enumerate(self.lateral_convs):
            laterals.append(conv(features[i+1]))
        
        for i in range(len(laterals) - 1):
            up = F.interpolate(laterals[i], size=laterals[i+1].shape[2:], mode='bilinear', align_corners=False)
            laterals[i+1] = laterals[i+1] + up
        
        outputs = []
        for i, conv in enumerate(self.smooth_convs):
            outputs.append(conv(laterals[i]))
        
        x = outputs[-1]

        x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        
        return x
    
class DeepLabV3Decoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super(DeepLabV3Decoder, self).__init__()
        
        self.aspp = ASPP(encoder_channels[-1])
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[1], 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, features):
        x = self.aspp(features[-1])
        
        x = F.interpolate(x, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        
        low_level_features = self.low_level_conv(features[1])
        
        x = torch.cat([x, low_level_features], dim=1)
        
        x = self.decoder(x)

        x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=False)

        x = self.final_conv(x)
        
        return x
    
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        out_channels = 256
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = self._make_aspp_conv(in_channels, out_channels, atrous_rates[0])
        self.conv3 = self._make_aspp_conv(in_channels, out_channels, atrous_rates[1])
        self.conv4 = self._make_aspp_conv(in_channels, out_channels, atrous_rates[2])
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def _make_aspp_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.size()[2:]
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        
        pool = self.global_avg_pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([conv1, conv2, conv3, conv4, pool], dim=1)
        
        x = self.out_conv(x)
        
        return x


