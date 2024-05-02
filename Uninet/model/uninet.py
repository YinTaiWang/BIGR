import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


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
        strides for each layer.
    upsample_kernel_size: list of tuples or int
        upsample_kernel_size for each layer stride[1:].
    filters: list, optional
        Specifies the number of filters for each layer.
    activation: str
        Type of activation to use.
    normalisation: str
        Type of normalisation (instance or batch).
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: tuple,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        activation: str = "LRELU",
        normalisation: str = "instance",
    ):
        super(UNet, self).__init__()
        
        self.name = "UNet"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size[::-1]
        self.filters = filters if filters else [32, 64, 128, 256, 512][:len(kernel_size)]
        Act = nn.LeakyReLU if activation == "LRELU" else (nn.ReLU if activation == "RELU" else nn.PReLU)
        Norm = nn.InstanceNorm3d if normalisation == "instance" else nn.BatchNorm3d
     
        # Encoder
        self.input_block = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for i, kernel in enumerate(self.kernel_size):
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = self.filters[i]
            # This is just for clarity in printing the network
            if i == 0:
                self.input_block = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=self.strides[i],
                    Norm=Norm,
                    Activation=Act,
                )
            elif i < len(self.kernel_size) - 1:
                self.encoders.append(
                    DoubleConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel,
                        stride=self.strides[i],
                        Norm=Norm,
                        Activation=Act,
                    )
                )
            else:
                self.encoders = nn.ModuleList(self.encoders)
                self.bottleneck = DoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=self.strides[i],
                    Norm=Norm,
                    Activation=Act,
                )
        
        # Decoders
        self.decoders1 = nn.ModuleList()
        self.decoders2 = nn.ModuleList()
        for i in range(len(self.filters)-1, 0, -1):
            self.decoders1.append(
                Up(
                in_channels=self.filters[i], 
                out_channels=self.filters[i-1], 
                kernel_size=kernel_size[i-1], 
                upsample_kernel_size=upsample_kernel_size[i], 
                Norm=Norm)
            )
            self.decoders2.append(
                Up(
                in_channels=self.filters[i], 
                out_channels=self.filters[i-1], 
                kernel_size=kernel_size[i-1], 
                upsample_kernel_size=upsample_kernel_size[i], 
                Norm=Norm)
            )

        self.final_conv1 = nn.Conv3d(self.filters[0], self.out_channels[0], 1)
        self.final_conv2 = nn.Conv3d(self.filters[0], self.out_channels[1], 1)
        
        self.weight_initializer()

    def forward(self, x):
        x = self.input_block(x)
        skips = [x]

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        x1 = x
        x2 = x
        for i, (dec1, dec2, skip) in enumerate(zip(self.decoders1, self.decoders2, reversed(skips))):
            x1 = dec1(x1, skip)
            x2 = dec2(x2, skip)
            if i == 0:
                # save the features from the 'i' decoder
                # collect these features for orthogonal regularization
                feature1 = x1  
                feature2 = x2
        
        x1 = self.final_conv1(x1)
        x2 = self.final_conv2(x2)

        return x1, x2, feature1, feature2
    
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
        kernel_size=3,
        stride=1,
        bias=True,
        Norm=nn.InstanceNorm3d,
        Activation=nn.LeakyReLU,
    ):
        super(DoubleConv, self).__init__()
        padding = kernel_size // 2  # Automatic padding calculation
        
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
        kernel_size,
        upsample_kernel_size,
        Norm=nn.InstanceNorm3d,
    ):
        super(Up, self).__init__()

        self.transpconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            bias=False,
        )
        self.doubleconv = DoubleConv(
            out_channels * 2, out_channels, kernel_size, stride=1, Norm=Norm
        )

    def forward(self, x, skip):
        x = self.transpconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate skip connection with upsampled output
        x = self.doubleconv(x)
        return x