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
    
    def __init__ (self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        '''
        LBO and SVD
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x, LBO_MATRIX, LBO_INVERSE, label = 'True'):
        
        # print('forward_BASE_MATRIX:', LBO_MATRIX.shape, 'BASE_INVERSE:', LBO_INVERSE.shape)              
        # LBO domain
        # print(label)
        if label == 'True':
            
            x = x = x.permute(0, 2, 1)
            x = LBO_INVERSE @ x  # LBO_MATRIX mesh_number*modes
            x = x.permute(0, 2, 1)
            
        if label == 'None':
            
            # print('此处运行！')
            # print(x.shape)
            x = LBO_INVERSE @ x  # LBO_MATRIX mesh_number*modes
            # print(x.shape)
            x = x.permute(0, 2, 1)
            # print(x.shape)
            
        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        
        # Back to x domain
        x =  x @ LBO_MATRIX.T
        
        return x
    
        
class MeshNO(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input):
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
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.fc01 = nn.Linear(self.width, self.width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        
        self.conv_encode = SpectralConv1d(self.width, self.width, self.modes1)
        
        # self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)
        
        # self.conv_decode = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        # self.w1 = nn.Conv1d(self.width, self.width, 1)
        # self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.fc0(x)
        x  = F.gelu(x)
        x = self.fc01(x)
        x = x.permute(0, 2, 1)
        
        x1 = self.conv_encode(x, self.LBO_Matri_input, self.LBO_Inver_input)
        x2 = self.w0(x)
        x  = x1 + x2
        x  = F.gelu(x)

        # x1 = self.conv0(x, self.LBO_Matri_input, self.LBO_Inver_input)
        # x2 = self.w1(x)
        # x  = x1 + x2
        # x  = F.gelu(x)
        
        x = self.conv1(x, self.LBO_Matri_output, self.LBO_Inver_input)
        x = F.gelu(x)
        
        # x1 = self.conv2(x, self.LBO_Matri_output, self.LBO_Inver_output)
        # x2 = self.w2(x)
        # x  = x1 + x2
        # x  = F.gelu(x)

        x1 = self.conv3(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w3(x)
        x  = x1  + x2
        
        x1 = self.conv4(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w4(x)
        x  = x1  + x2

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
