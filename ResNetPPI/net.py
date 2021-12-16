# Copyright 2021 Zefeng Zhu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Created Date: 2021-12-14 06:44:56 pm
# @Filename: net.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-16 12:18:49 am
import torch.nn as nn


class ResidualBlockBase(nn.Module):
    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class ResNetLayerBase(nn.Module):
    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet1DResidualBlock(ResidualBlockBase):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super().__init__()
        padding = dilation*((kernel_size-1)//2)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    dilation=dilation, padding=padding, bias=False, **kwargs),
                          nn.BatchNorm1d(out_channels)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    dilation=dilation, padding=padding, bias=False, **kwargs),
                          nn.BatchNorm1d(out_channels))
        )
        self.activate = nn.ELU(inplace=True)


class ResNet2DResidualBlock(ResidualBlockBase):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        padding = (dilation*(kernel_size-1)//2, dilation*(kernel_size-1)//2)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    dilation=dilation, padding=padding, bias=False, **kwargs),
                          nn.BatchNorm2d(out_channels)),
            nn.ELU(inplace=True),
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    dilation=dilation, padding=padding, bias=False, **kwargs),
                          nn.BatchNorm2d(out_channels)),
        )
        self.activate = nn.ELU(inplace=True)


class ResNet1DLayer(ResNetLayerBase):
    def __init__(self, out_channels, kernel_size, dilation, n_layer, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResNet1DResidualBlock(out_channels, out_channels, kernel_size, dilation, **kwargs) for _ in range(n_layer)]
        )


class ResNet2DLayer(ResNetLayerBase):
    def __init__(self, out_channels, kernel_size, dilation, n_layer, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResNet2DResidualBlock(out_channels, out_channels, kernel_size, dilation, **kwargs) for _ in range(n_layer)]
        )


class ResNet1D(nn.Module):
    def __init__(self, in_channels, deepths, kernel_size=3, channel_size=64, dilation=1, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, channel_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(channel_size),
            nn.ELU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            *[ResNet1DLayer(channel_size, kernel_size, dilation, n_layer=n, **kwargs) for n in deepths]
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNet2D(nn.Module):
    def __init__(self, in_channels, deepths, kernel_size=3, channel_size=96, dilation=2, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, channel_size, kernel_size=kernel_size, padding=2, bias=False),
            nn.BatchNorm2d(channel_size),
            nn.ELU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            *[ResNet2DLayer(channel_size, kernel_size, dilation, n_layer=n, **kwargs) for n in deepths]
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
