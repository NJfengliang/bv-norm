import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################
#
# Utilities
#
#################################################


################################################################
#  1d fourier layer
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
            # nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            # nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            # nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            # nn.GELU(),
        )

    # nn.Sequential()为时序容器，模型会以传入的顺序被添加进容器
    def forward(self, x):
        return self.double_conv(x)



class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

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


class FNO1d(nn.Module):
    def __init__(self, channel, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, i=x)
        output: the solution of a later timestep
        output shape: (batchsize, c=modes)
        """
        self.modes1 = 32
        self.modes = channel
        self.width = width
        self.channel = channel
        self.padding = 2  # pad the domain if input is non-periodic
        # self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)


        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv5 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        # self.w0 = DoubleConv(self.width, self.width)
        # self.w1 = DoubleConv(self.width, self.width)
        # self.w2 = DoubleConv(self.width, self.width)
        # self.w3 = DoubleConv(self.width, self.width)
        # self.w4 = nn.Conv1d(self.width, self.width, 1)
        # self.w5 = nn.Conv1d(self.width, self.width, 1)

        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)
        # # self.fc4 = nn.Linear(8576, self.channel)
        # #
        # self.out = nn.Conv1d(151, self.modes, kernel_size=3, padding=1)
        # self.inp = nn.Conv1d(2, self.width, kernel_size=1)
        # self.inp2 = nn.Linear(2, self.width)

        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 1)
        # # self.out = nn.Conv1d(151, 1, 1)
        # self.out = nn.Linear(151, self.channel)
        self.fc3 = nn.Linear(self.width, 1)
        # self.fc4 = nn.Linear(101, self.channel)
        self.out = nn.Conv1d(151, self.channel,1)
        

    def forward(self, x):
        # x = x.unsqueeze(2)
          # x0
        ##############################   Fourier Residual Layer #################
        # x_in_ft = torch.fft.rfft(x_in1, axis=-2)
        # x_in_ft[:, self.modes1:, :] = 0
        # x_ifft = torch.fft.irfft(x_in_ft, n=x_in1.size(-2), axis=-2)
        ########################################################################

        # x = self.fc0(x)  # x1
        # x_in2 = x

        # x = x.permute(0, 2, 1)  #
        # x = self.inp(x)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)  # x2
        # x = self.drop(x)
        # x_in1 = x

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)  # x3
        # x = self.drop(x)
######################################
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv4(x)
        # x2 = self.w4(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv5(x)
        # x2 = self.w5(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x = self.drop(x)  # x4
#########################################
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2 #+ x_in1 # x5

        x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        x = self.out(x)
        x = F.gelu(x)  # x6
        # x = self.drop(x)

        #x = self.fc2(x) #+ x_in2  # x7
        # x = self.out(x)
        # x = self.fc1(x)  # x8
        # x = self.fc2(x)
        # x = self.fc3(x)

        # x = self.fc2(x) #+ x_in2  # 10x151x64
        # x = self.fc3(x)  # 10x151x8
        # x = torch.squeeze(x)
        # x = self.out(x)


        # x = torch.squeeze(x)
        #x = self.fc4(x)
        x = self.fc3(x)  # x8

        x = torch.squeeze(x)

        ##############################   Fourier Residual Layer #################
        # x = x  + x_ifft 
        ########################################################################

        return x
