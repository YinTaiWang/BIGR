import warnings
from typing import Optional, Sequence

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
    out_channels: int
        Number of output channels.
    channels: list, optional
        Specifies the number of channels for each layer.
    act: str
        Type of activation to use.
    norm: str
        Type of normalization (instance or batch).
    '''
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Optional[Sequence[int]] = (32, 64, 128, 256),
        act: str = "LRELU",
        norm: str = "instance",
    ):
        super(UNet, self).__init__()
        
        assert spatial_dims == 3, 'Currently only 3D is possible.'
        if len(channels) < 2:
            raise ValueError("the length of 'channels' should be no less than 2.")
        
        self.name = "UNet"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.act = nn.LeakyReLU if act == "LRELU" else (nn.ReLU if act == "RELU" else nn.PReLU)
        self.norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        
        # Encoder
        self.input_block = DoubleConv(
            in_channels=in_channels,
            out_channels=channels[0],
            stride=1,
            Norm=self.norm,
            Activation=self.act,
        )
        self.encoders = nn.ModuleList()
        for i in range(1, len(self.channels) - 1):
            self.encoders.append(
                DoubleConv(
                    in_channels=channels[i-1],
                    out_channels=channels[i],
                    stride=2,
                    Norm=self.norm,
                    Activation=self.act,
                )
            )
        self.bottleneck = DoubleConv(
            in_channels=channels[-2],
            out_channels=channels[-1],
            stride=2,
            Norm=self.norm,
            Activation=self.act,
        )
        
        # Decoders
        self.decoders = nn.ModuleList()
        for i in range(len(self.channels)-1, 0, -1):
            self.decoders.append(
                Up(
                    in_channels=self.channels[i], 
                    out_channels=self.channels[i-1],
                    stride=2,
                    Norm=self.norm
                )
            )

        self.final_conv = nn.Conv3d(self.channels[0], self.out_channels, 1)
        
        self.weight_initializer()
    
    def forward(self, x):
        x = self.input_block(x)
        skips = [x]

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        
        x = self.bottleneck(x)

        for (dec, skip) in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
            
        x = self.final_conv(x)
        
        return x
    
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
    
    
##########################################################################
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        bias=True,
        Norm=nn.InstanceNorm3d,
        Activation=nn.LeakyReLU,
    ):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm1 = Norm(out_channels, affine=True)
        self.act1 = Activation()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
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
        stride,
        Norm=nn.InstanceNorm3d,
    ):
        super(Up, self).__init__()

        self.transpconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=stride,
            bias=False,
        )
        self.doubleconv = DoubleConv(
            out_channels * 2, out_channels, stride=1, Norm=Norm
        )

    def forward(self, x, skip):
        x = self.transpconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate skip connection with upsampled output
        x = self.doubleconv(x)
        return x
