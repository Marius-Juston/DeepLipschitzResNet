from torch import nn
from torch.nn import ZeroPad2d

from models.cnn_model import DeepLipschitzConvResNet
from models.linear_model import DeepLipschitzLinearResNet


class LipschitzNetwork(nn.Module):

    def __init__(self, config, n_classes):
        super(LipschitzNetwork, self).__init__()

        self.n_conv: int = config.n_conv
        self.n_interior_conv: int = config.n_interior_conv
        self.n_dense: int = config.n_dense
        self.n_interior_dense: int = config.n_interior_dense
        self.cin: int = config.w
        self.conv_inner_dim: int = config.conv_inner_dim
        self.dense_inner_dim: int = config.dense_inner_dim
        self.n_classes: int = n_classes

        if config.dataset == 'tiny-imagenet':
            imsize = 64
        else:
            imsize = 32

        self.model = []
        self.model.append(
            ZeroPad2d(3)
        )

        inner_conv_layers = [self.conv_inner_dim] * self.n_interior_conv

        for _ in range(self.n_conv):
            self.model.append(DeepLipschitzConvResNet(self.cin, self.conv_inner_dim, inner_conv_layers))

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
            self.model.append(DeepLipschitzLinearResNet(in_channels, self.dense_inner_dim, inner_linear_layers))

        self.model.append(DeepLipschitzLinearResNet(in_channels, self.n_classes, inner_linear_layers))

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
