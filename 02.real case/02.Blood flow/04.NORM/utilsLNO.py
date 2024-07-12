# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:23:56 2022

@author: GengxiangCHEN
"""

import torch
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#LBO Network
#########################
class SpectralConv1d(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, LBO_MATRIX, LBO_INVERSE):
        super(SpectralConv1d, self).__init__()
        '''
        LBO and SVD
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = LBO_MATRIX.shape[1]
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
                        
        # LBO domain
        x = x = x.permute(0, 2, 1)
        x = self.LBO_INVERSE @ x  # LBO_MATRIX mesh_number*modes
        x = x.permute(0, 2, 1)
        
        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        
        # Back to x domain
        x =  x @ self.LBO_MATRIX.T
        
        return x


class SpectralF1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralF1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1/2)  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # Tensor operation
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier modes
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Abandon high frequency modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x

class SpectralM1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,basis):
        super(SpectralM1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = int(modes1/2)  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.basis = basis

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # Tensor operation
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier modes
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Abandon high frequency modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        # x = torch.fft.irfft(out_ft, n=x.size(-1))
        # x = x @ self.basis.T
        # out = torch.real(out_ft)
        out_r = torch.real(out_ft[:, :, :self.modes1])
        out_i = torch.imag(out_ft[:, :, :self.modes1])
        out = torch.cat((out_r, out_i), 2)
        # out_i = out_i.unsqueeze(3)
        # out_r = out_r.unsqueeze(3)
        # x = out[:, :, :self.modes1]@ self.basis.T
        # out = torch.cat((out_r, out_i), 3)
        # out = out.reshape(-1,self.in_channels,2*self.modes1)

        x = out @ self.basis.T
        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
            # nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            # nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            # nn.GELU(),
        )

    # nn.Sequential()为时序容器，模型会以传入的顺序被添加进容器
    def forward(self, x):
        return self.double_conv(x)
    
        
class MeshNO(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE,E,E_inverse):
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
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(7, self.width) # input channel is 2: (a(x), x)
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE
        self.LBO1D_MATRIX = E
        self.LBO1D_INVERSE = E_inverse
        self.nodes = self.LBO_MATRIX.shape[0]

        self.conv0 = SpectralF1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralM1d(self.width, self.width, self.modes1, self.LBO_MATRIX)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )

        # self.conv0 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO1D_MATRIX, self.LBO1D_INVERSE)
        # self.conv1 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO1D_INVERSE)
        # self.conv2 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        # self.conv3 = SpectralConv1d(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # self.out = DoubleConv(202, self.nodes)

        self.fc1 = nn.Linear(self.width, self.modes1)
        self.fc2 = nn.Linear(self.modes1, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        # x = self.out(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x1)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     
