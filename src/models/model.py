import logging
from argparse import Namespace
from typing import Sequence, Tuple, Optional

import numpy as np
import torch.nn
import torch.nn.functional as F
from torch import nn

from models.batch_norm import LipschitzBatchNorm
from models.cnn_model import DeepLipschitzConvResNet, DeepLipschitzConvSequential
from models.linear_model import DeepLipschitzSequential

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
}


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


def is_complex_config(config: Namespace):
    if (isinstance(config.n_conv, Sequence) and
            isinstance(config.n_conv, Sequence) and
            config.dense_inner_dim is None and
            config.conv_inner_dim is None):
        return True
    elif (isinstance(config.n_conv, int) and
          isinstance(config.n_dense, int) and
          isinstance(config.dense_inner_dim, int)
          and isinstance(config.conv_inner_dim, int)):
        return False
    else:
        raise RuntimeError(
            "The config is not properly setup for the Lipschitz network configuration, it must either be the complex or the simple setup")


class LipschitzNetwork(nn.Module):
    def generate_simple(self, config: Namespace):
        n_conv: int = config.n_conv
        n_dense: int = config.n_dense
        cin: int = config.w
        conv_inner_dim: int = config.conv_inner_dim
        dense_inner_dim: int = config.dense_inner_dim
        n_classes: int = self.n_classes
        batch_norm: bool = config.batch_norm
        dropout: bool = config.dropout

        model = []
        model.append(
            PaddingChannels(cin, 3, 'zero')
        )

        inner_conv_layers = [conv_inner_dim] * n_conv

        if hasattr(config, "activation") and config.activation in ACTIVATION_FUNCTIONS:
            activation = ACTIVATION_FUNCTIONS[config.activation]()
        else:
            activation = torch.nn.ReLU()

        if batch_norm:
            model.append(LipschitzBatchNorm(cin, scale=True, device=self.device))

        model.append(
            self.conv_model(cin, cin, inner_conv_layers, use_safe_cholesky=config.safe_cholesky,
                            wrong_normalization=config.wrong_normalization,
                            activation=activation,
                            dropout=dropout,
                            device=self.device))

        model.append(nn.AvgPool2d(4, divisor_override=4))

        model.append(nn.Flatten())

        inner_linear_layers = [dense_inner_dim] * n_dense

        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = cin * 16 * 16
        else:
            raise ValueError("The dataset must be 'cifar10', 'cifar100' or 'tiny-imagenet'")

        if batch_norm:
            model.append(LipschitzBatchNorm(in_channels, scale=True, device=self.device))

        model.append(
            DeepLipschitzSequential(in_channels, n_classes, inner_linear_layers, activation=activation, dropout=dropout,
                                    device=self.device))

        return model

    def generate_complex(self, config):
        n_conv: Sequence[Tuple[int, int, int]] = config.n_conv
        n_dense: Sequence[Tuple[int, int]] = config.n_dense

        model = []

        cin = 3

        inner_conv_layers = []

        last_channel_size = cin

        for n, channels, kernel in n_conv:
            for _ in range(n):
                inner_conv_layers.append([channels, kernel])

            last_channel_size = channels

        logging.info(f"Setting up model with conv config: {inner_conv_layers}")
        logging.info(f"Cin: {cin}")
        logging.info(f"Last channels: {last_channel_size}")

        if hasattr(config, "activation") and config.activation in ACTIVATION_FUNCTIONS:
            activation = ACTIVATION_FUNCTIONS[config.activation]()
        else:
            activation = torch.nn.ReLU()

        model.append(
            self.conv_model(cin, last_channel_size, inner_conv_layers, use_safe_cholesky=config.safe_cholesky,
                            device=self.device, activation=activation, use_averaging=config.averaging))

        model.append(nn.AvgPool2d(4, divisor_override=4))

        model.append(nn.Flatten())

        inner_dense_layers = []

        for n, outputs in n_dense:
            for _ in range(n):
                inner_dense_layers.append(outputs)

        if config.dataset in ['cifar10', 'cifar100']:
            in_channels = cin * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = cin * 16 * 16
        else:
            raise ValueError("The dataset must be 'cifar10', 'cifar100' or 'tiny-imagenet'")

        in_channels = 16384  # hardcoded for this current task should not be hard-coded

        model.append(DeepLipschitzSequential(in_channels, self.n_classes, inner_dense_layers, device=self.device,
                                             activation=activation))

        return model

    def __init__(self, config: Namespace, n_classes: int, device: Optional[str] = None):
        super(LipschitzNetwork, self).__init__()

        self.device = device
        self.n_classes = n_classes
        self.config = config

        if config.linear_conv:
            self.conv_model = DeepLipschitzConvSequential
        else:
            self.conv_model = DeepLipschitzConvResNet
        logging.info(f"Using model ({config.linear_conv}): {self.conv_model.__name__}")

        if is_complex_config(config):
            logging.info("Using complex configuration for model setup")
            model = self.generate_complex(config)
        else:
            logging.info("Using simple configuration for model setup")
            model = self.generate_simple(config)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
