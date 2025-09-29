import numpy as np
import torch.nn.functional as F
from torch import nn

from models.cnn_model import DeepLipschitzConvResNet
from models.linear_model import DeepLipschitzLinearResNet


class PaddingChannels(nn.Module):
    def __init__(self, ncout, ncin=3, mode="zero"):
        super(PaddingChannels, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

        assert mode in ["zero", "clone"], "You must have the mode either be 'zero' or 'clone'"

        self.pad_channels = self.ncout - self.ncin

        self.repeat_num = int(self.ncout / self.ncin)
        self.repeat_norm = np.sqrt(int(self.ncout / self.ncin))

    def forward(self, x):
        if self.mode == "zero":
            return F.pad(x, (0, 0, 0, 0, 0, self.pad_channels), mode="constant", value=0)
        elif self.mode == "clone":
            return x.repeat(1, self.repeat_num, 1, 1) / self.repeat_norm
        return None


class LipschitzNetwork(nn.Module):

    def __init__(self, config, n_classes, device=None):
        super(LipschitzNetwork, self).__init__()

        self.n_conv: int = config.n_conv
        self.n_interior_conv: int = config.n_interior_conv
        self.n_dense: int = config.n_dense
        self.n_interior_dense: int = config.n_interior_dense
        self.cin: int = config.w
        self.conv_inner_dim: int = config.conv_inner_dim
        self.dense_inner_dim: int = config.dense_inner_dim
        self.n_classes: int = n_classes
        self.dense_next_layer: int = config.dense_next_layer

        if config.dataset == 'tiny-imagenet':
            imsize = 64
        else:
            imsize = 32

        self.model = []
        self.model.append(
            PaddingChannels(self.cin, 3, 'zero')
        )

        inner_conv_layers = [self.conv_inner_dim] * self.n_interior_conv

        for _ in range(self.n_conv):
            self.model.append(
                DeepLipschitzConvResNet(self.cin, self.cin, inner_conv_layers, use_safe_cholesky=config.safe_cholesky,
                                        device=device))

        self.model.append(nn.AvgPool2d(4, divisor_override=4))

        self.model.append(nn.Flatten())
        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = self.cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = self.cin * 16 * 16
        else:
            raise ValueError("The dataset must be 'cifar10', 'cifar100' or 'tiny-imagenet'")

        inner_linear_layers = [self.conv_inner_dim] * self.n_interior_conv

        for _ in range(self.n_dense):
            self.model.append(
                DeepLipschitzLinearResNet(in_channels, self.dense_next_layer, inner_linear_layers,
                                          use_safe_cholesky=config.safe_cholesky, device=device))

            in_channels = self.dense_next_layer

        self.model.append(DeepLipschitzLinearResNet(in_channels, self.n_classes, inner_linear_layers,
                                                    use_safe_cholesky=config.safe_cholesky, device=device))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
