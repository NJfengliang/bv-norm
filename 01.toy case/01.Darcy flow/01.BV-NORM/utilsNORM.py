# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:23:56 2022

@author: GengxiangCHEN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# LBO Network
#########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, LBO_MATRIX, LBO_INVERSE):
        super(SpectralConv1d, self).__init__()
        '''
        LBO 
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = LBO_MATRIX.shape[1]
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
        # LBO domain
        x = x.permute(0, 2, 1)
        x = self.LBO_INVERSE @ x  # LBO_MATRIX mesh_number*modes
        x = x.permute(0, 2, 1)

        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)

        # Back to x domain
        x = x @ self.LBO_MATRIX.T

        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )

    # nn.Sequential()为时序容器，模型会以传入的顺序被添加进容器
    def forward(self, x):
        return self.double_conv(x)


class MeshNO(nn.Module):
    def __init__(self, modes, LBO_MATRIX, LBO_INVERSE, width):
        super(MeshNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE
        self.nodes = LBO_MATRIX.shape[0]

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # self.w0 = DoubleConv(self.width, self.width)
        # self.w1 = DoubleConv(self.width, self.width)
        # self.w2 = DoubleConv(self.width, self.width)
        # self.w3 = DoubleConv(self.width, self.width)

        self.fc = nn.Linear(self.width, 1)
        self.fc1 = nn.Linear(2, self.modes1)
        self.out = nn.Conv1d(2, self.modes1, kernel_size=3, padding=1)
        # self.out2 = DoubleConv(2, self.modes1)

    def forward(self, x):

        grid = self.get_grid(x.shape, x.device)
        x = x.unsqueeze(2)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(1, 0, 2)
        x = self.out(x)
        x = x.permute(1, 2, 0)

        ## NORM layer
        # ————————————————————————————————————————————————————————————————
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        # ————————————————————————————————————————————————————————————————

        ## output layer
        x = x.permute(2, 0, 1)
        x = self.fc(x)
        x = torch.squeeze(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
