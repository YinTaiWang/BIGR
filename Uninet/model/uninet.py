import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    '''
    UNet. 
    Adapted from: https://github.com/Douwe-Spaanderman/InteractiveNet/blob/main/interactivenet/networks/unet.py

    Parameters:
    in_channels: int
        Number of channels in the input tensor.
    out_channels: tuple
        Tuple of the number of output channels for each decoder.
    kernel_size: list of tuples or int
        kernel_sizes for each layer. Also determines number of layer
    strides: list of tuples or int
        strides for each layer. The length of 'strides' should equal to 'len(channels) - 1'.
    filters: list, optional
        Specifies the number of filters for each layer.
    activation: str
        Type of activation to use.
    normalisation: str
        Type of normalisation (instance or batch).
    '''
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: tuple,
        channels: Optional[Sequence[int]] = (32, 64, 128, 256),
        strides: Optional[Sequence[int]] = (2, 2, 2),
        kernel_size: Optional[Sequence[int]] = (3, 3, 3),
        up_kernel_size: Optional[Sequence[int]] = (3, 3, 3),
        act: str = "LRELU",
        norm: str = "instance",
    ):
        super(UNet, self).__init__()
        
        assert spatial_dims == 3, 'Currently only 3D is possible.'
        if len(channels) < 2:
            raise ValueError("the length of 'channels' should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of 'strides' should equal to 'len(channels) - 1'.")
        if delta > 0:
            warnings.warn(f"'len(strides) > len(channels) - 1', the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of 'kernel_size' should equal to dimension, 3.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of 'up_kernel_size' should equal to dimension, 3.")
        
        self.name = "UNet"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = nn.LeakyReLU if act == "LRELU" else (nn.ReLU if act == "RELU" else nn.PReLU)
        self.norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        
        # Encoder
        self.input_block = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        for i, channel in enumerate(self.channels):
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = channel
            # Input block
            if i == 0:
                self.input_block = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    Norm=self.norm,
                    Activation=self.act,
                )
            # Downsampling 1
            elif i < len(self.channels) - 1:
                self.encoders.append(
                    DoubleConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.strides[i-1],
                        Norm=self.norm,
                        Activation=self.act,
                    )
                )
            # Bottleneck
            else:
                self.encoders = nn.ModuleList(self.encoders)
                self.bottleneck = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.strides[i-1],
                    Norm=self.norm,
                    Activation=self.act,
                )
        
        # Decoders
        self.decoders1 = nn.ModuleList()
        self.decoders2 = nn.ModuleList()
        for i in range(len(self.channels)-1, 0, -1):
            self.decoders1.append(
                Up(
                in_channels=self.channels[i], 
                out_channels=self.channels[i-1], 
                upsample_kernel_size=self.up_kernel_size, 
                stride=self.strides[i-1],
                Norm=self.norm)
            )
            self.decoders2.append(
                Up(
                in_channels=self.channels[i], 
                out_channels=self.channels[i-1], 
                upsample_kernel_size=self.up_kernel_size, 
                stride=self.strides[i-1],
                Norm=self.norm)
            )

        self.final_conv1 = nn.Conv3d(self.channels[0], self.out_channels[0], 1)
        self.final_conv2 = nn.Conv3d(self.channels[0], self.out_channels[1], 1)
        
        self.weight_initializer()
    
    def forward(self, x):
        x = self.input_block(x)
        print("encode", x.shape)
        skips = [x]

        for enc in self.encoders:
            x = enc(x)
            print("encode", x.shape)
            skips.append(x)

        x = self.bottleneck(x)
        print("bottomshape", x.shape)

        x1 = x
        x2 = x
        for i, (dec1, dec2, skip) in enumerate(zip(self.decoders1, self.decoders2, reversed(skips))):
            x1 = dec1(x1, skip)
            x2 = dec2(x2, skip)
            print("decode 1", x1.shape)
            print("decode 2", x2.shape)
        #     if i == 0:
        #         # save the features from the 'i' decoder
        #         # collect these features for orthogonal regularization
        #         feature1 = x1  
        #         feature2 = x2
        
        # ortho_loss = self.ortho(feature1, feature2)
        # print(f"Orthogonal loss: {ortho_loss.item()}")
        
        x1 = self.final_conv1(x1)
        x2 = self.final_conv2(x2)
        
        return x1, x2
    
    def weight_initializer(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, a=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d) or isinstance(
                module, nn.BatchNorm1d
            ):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def ortho(self, feature1, feature2):
        # Assuming features are [batch_size, num_features, height, width, depth]
        feature1_flat = feature1.view(feature1.size(0), -1)
        feature2_flat = feature2.view(feature2.size(0), -1)

        # Normalize the feature vectors to have unit length
        feature1_norm = F.normalize(feature1_flat, p=2, dim=1)
        feature2_norm = F.normalize(feature2_flat, p=2, dim=1)

        # Calculate the cosine similarity and minimize its square
        cosine_similarity = torch.mm(feature1_norm, feature2_norm.t())
        ortho_loss = cosine_similarity.pow(2).mean()

        return ortho_loss
                    
                    
##########################################################################
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=True,
        Norm=nn.InstanceNorm3d,
        Activation=nn.LeakyReLU,
    ):
        super(DoubleConv, self).__init__()
        if kernel_size == 3 or kernel_size == (3, 3, 3):
            padding = 1
        elif kernel_size == (3, 3, 1):
            padding = (1, 1, 0)
        elif kernel_size == (1, 3, 3):
            padding = (0, 1, 1)
        else:
            padding = 1
            warnings.warn(
                "kernel is neither 3, (3,3,3) or (1,3,3). This scenario has not been correctly implemented yet, but using padding = 1"
            )
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm1 = Norm(out_channels, affine=True)
        self.act1 = Activation()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.norm2 = Norm(out_channels, affine=True)
        self.act2 = Activation()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_kernel_size,
        stride,
        Norm=nn.InstanceNorm3d,
    ):
        super(Up, self).__init__()

        self.transpconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=stride,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.doubleconv = DoubleConv(
            out_channels * 2, out_channels, upsample_kernel_size, stride=1, Norm=Norm
        )

    def forward(self, x, skip):
        x = self.transpconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate skip connection with upsampled output
        x = self.doubleconv(x)
        return x