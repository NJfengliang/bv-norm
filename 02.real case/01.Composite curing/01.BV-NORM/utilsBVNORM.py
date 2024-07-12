# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
            # nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            # nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            # nn.GELU(),
        )

    # nn.Sequential()为时序容器，模型会以传入的顺序被添加进容器
    def forward(self, x):
        return self.double_conv(x)


class BVNORM(nn.Module):

    def __init__(
            self,
            basis,
            BC_NET,
            Geo_NET=None,

    ):
        super().__init__()

        self.basis = basis
        self.modes = basis.shape[1]
        self.bc = BC_NET
        self.geo = None
        if Geo_NET is not None:
            self.geo = Geo_NET
        # self.b = nn.parameter.Parameter(torch.tensor(0.0))
        # self.out = DoubleConv(2, 1)

    def forward(self, inputs):
        x_func = inputs[0]
        ##reshape
        # x_func = self.out(x_func)
        # x_func = torch.squeeze(x_func)
        ####
        x_loc = inputs[1]
        x_func = self.bc(x_func).to(device)
        if self.geo is None:
            # basis only
            if isinstance(x_func, torch.Tensor) and x_func.dim() == 2:

                x = torch.einsum("bm,xm->bx", x_func, self.basis)
                # x = torch.einsum("bm,xm->bx", x_func, x1)

            else:
                x = torch.einsum("m,xm->x", x_func, self.basis)
                # x = torch.einsum("m,xm->x", x_func, x1)
            # x += self.bias
        else:
            x_loc = self.geo(x_loc)
            if isinstance(x_func, torch.Tensor) and x_func.dim() == 2:
                x = torch.einsum("bm,xm->bx", x_func, x_loc)
            else:
                x = torch.einsum("m,xm->x", x_func, x_loc)
            # x += self.b

        return x
