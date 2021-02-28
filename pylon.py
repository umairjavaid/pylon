import torch
import torch.nn.functional as F
from torch import nn

from segmentation_models_pytorch.base import (ClassificationHead,
                                              SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.pan.decoder import ConvBnRelu
import copy
heatmap = torch.Tensor(1)

class Pylon(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 pretrain='imagenet',
                 n_dec_ch=128,
                 n_in=1,
                 n_out=14):
        super(Pylon, self).__init__()
        self.net = PylonCore(
            encoder_name=backbone,
            encoder_weights=pretrain,
            decoder_channels=n_dec_ch,
            in_channels=n_in,
            classes=n_out,
            upsampling=1,
            align_corners=True,
        )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.net(x).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

        return {
            'pred': pred,
            'seg': seg,
        }


class PylonCore(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = "imagenet",
            decoder_channels: int = 128,
            in_channels: int = 1,
            classes: int = 1,
            upsampling: int = 1,
            align_corners=True,
    ):
        super(PylonCore, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        self.decoder = PylonDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=classes,
            upscale_mode='bilinear',
            align_corners=align_corners,
        )

        self.segmentation_head = SegmentationHead(in_channels=classes,
                                                  out_channels=classes,
                                                  activation=None,
                                                  kernel_size=1,
                                                  upsampling=upsampling)

        # just to comply with SegmentationModel
        self.classification_head = None

        self.name = "pylon-{}".format(encoder_name)
        self.initialize()

class PylonDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, upscale_mode='bilinear', align_corners=True,):
            super(PylonDecoder, self).__init__()
            self.conv6 = nn.Conv2d(encoder_channels[-1],  1024, kernel_size=3, padding=1) 
            self.conv7 = nn.Conv2d(1024, decoder_channels, kernel_size=1)
            self.conv8 = nn.Conv2d(encoder_channels[-1],  1024, kernel_size=3, padding=1) 
            self.conv9 = nn.Conv2d(1024, decoder_channels, kernel_size=1)
            self.conv10 = nn.Conv2d(encoder_channels[-1],  1024, kernel_size=3, padding=1) 
            self.conv11 = nn.Conv2d(1024, decoder_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=False)
            
    def forward(self, *x):
            x1 = self.conv6(x[-1])
            x1 = self.relu(x1)
            x1 = self.conv7(x1)
            x1 = self.relu(x1)
        
            x2 = self.conv8(x[-1])
            x2 = self.relu(x2)
            x2 = self.conv9(x2)
            x2 = self.relu(x2)

            x3 = self.conv10(x[-1])
            x3 = self.relu(x3)
            x3 = self.conv11(x3)
            x3 = self.relu(x3)
            #y = x1 + x2 + x3
            #global heatmap
            #heatmap.data = y.clone().float()
            x = torch.max(x1 ,x2)
            x = torch.max(x ,x3)
            return x

