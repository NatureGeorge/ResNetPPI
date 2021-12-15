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
# @Last Modified: 2021-12-14 06:45:10 pm
import torch.nn as nn


class ResNet1DResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        if in_channels != out_channels:
            self.projection = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                                                      dilation=dilation, padding=padding, bias=False, **kwargs),
                                            nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = x
        x = self.blocks(x)
        if self.in_channels != self.out_channels:
            residual = self.projection(residual)
        x += residual
        x = self.activate(x)
        return x


class ResNet2DResidualBlock(nn.Module):
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
    
    def forward(self, x):
        residual = x
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class ResNet2DLayer(nn.Module):
    def __init__(self, out_channels, kernel_size, dilation, n_layer, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResNet2DResidualBlock(out_channels, out_channels, kernel_size, dilation, **kwargs) for _ in range(n_layer)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet2D(nn.Module):
    def __init__(self, in_channels, deepths, 
                        kernel_size=3, channel_size=96, dilation=2,
                        **kwargs):
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
