import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def forward(self, inputs):
        x = inputs
        for linear_layer in self.linears:
            x = linear_layer(x)
        return x


class CNN(nn.Module):

    def __init__(self, layer_sizes, kernel_size, padding):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.conv_layers.append(
                torch.nn.Conv1d(
                    layer_sizes[i - 1], layer_sizes[i], kernel_size, padding=padding
                ).to(device=device)
            )

    def forward(self, x):
        x_in = x
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            x = x.unsqueeze(2)
        x = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.permute(0, 2, 1)
        x = torch.squeeze(x)
        x = x + x_in
        return x


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x
