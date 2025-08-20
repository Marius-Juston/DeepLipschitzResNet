import math
import os
from datetime import datetime
from typing import Sequence, Union, Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, ModuleList, ReLU, MSELoss, Parameter, init
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')


# @torch.compile
def matrix_inv_square_root(A: Tensor) -> Tensor:
    eigvals, eigvecs = torch.linalg.eigh(A.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(A):
    L = torch.linalg.cholesky_ex(A).L
    return torch.cholesky_inverse(L)


class LipschitzInnerLayer(Linear):
    __constants__ = ['identity', 'identity_out', 'constant', 'constant_sq']

    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constant: float = 1.0,
                 device=None,
                 dtype=None):
        super(LipschitzInnerLayer, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "identity", torch.eye(in_features, **factory_kwargs)
        )

        self.register_buffer(
            "identity_out", torch.eye(out_features, **factory_kwargs)
        )

        self.register_buffer(
            "constant", torch.tensor(constant, **factory_kwargs)
        )

        self.register_buffer(
            "constant_sq", torch.tensor(constant * constant, **factory_kwargs)
        )

        self.identity: Optional[Tensor]
        self.constant: Optional[Tensor]
        self.constant_sq: Optional[Tensor]

    def compute_weight(self, scaling: Optional[Tensor] = None) -> Tensor:
        inner = self.identity + self.weight.T @ self.weight
        inner_inv_sqrt = matrix_inv_square_root(inner)

        weight = self.constant * self.weight @ inner_inv_sqrt

        if scaling is not None:
            weight = weight @ scaling

        return weight

    def compute_d(self):
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(
            self.identity_out + self.weight @ self.weight.T)

    def compute_d1(self):
        return self.constant_sq * fast_symmetric_positive_definitive_matrix_inverse(
            self.identity + self.weight.T @ self.weight)

    def compute_d1_inv(self):
        return (self.identity + self.weight.T @ self.weight) / self.constant_sq

    def compute_d1_sqrt(self):
        return self.constant * matrix_inv_square_root(self.identity + self.weight.T @ self.weight)

    def compute_d_sqrt(self):
        return self.constant * matrix_inv_square_root(self.identity_out + self.weight @ self.weight.T)

    def compute_d_inv(self):
        return matrix_inv_square_root(self.identity_out + self.weight @ self.weight.T) / self.constant_sq

    def forward(self, input: Tensor, scaling: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        weight = self.compute_weight(scaling)

        return F.linear(input, weight, self.bias), weight


class DeepLipschitzResNet(Module):
    A: LipschitzInnerLayer
    B: LipschitzInnerLayer
    layers: ModuleList

    def __init__(self, in_features: int, out_features: int, layers_widths: Sequence[int],
                 activation: Union[Module, List[Module]] = None, lipschitz_constant: float = 1.0, bias: bool = True,
                 device: Union[str, torch.device] = 'cuda', dtype=None) -> None:
        super().__init__()

        assert len(layers_widths) >= 1, "This deep network assumes that the number of layers is greater "
        assert in_features > 0, "The input width needs to be larger than 0"
        assert out_features > 0, "The output width needs to be larger than 0"

        self.lipschitz_constant = lipschitz_constant
        self.out_features = out_features
        self.in_features = in_features
        self.layer_widths = layers_widths

        self.dtype = dtype
        self.device = device

        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.sqrt_2 = 1.4142135623730950  # the value of sqrt(2) to 16 decimal places

        if activation is None:
            self.activation = ReLU()
        else:
            self.activation = activation

        if bias:
            self.bias = Parameter(torch.empty(out_features, **self.factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.generate_layers()

        self.reset_bias()

    def reset_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.A.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self) -> None:
        self.reset_bias()

        # Find a better initialization scheme for the current formulation of the system
        self.A.reset_parameters()
        self.B.reset_parameters()

        for layer in self.layers:
            layer.reset_parameters()

    def generate_layers(self) -> None:
        self.A = LipschitzInnerLayer(in_features=self.in_features, out_features=self.out_features,
                                     constant=self.lipschitz_constant, **self.factory_kwargs)

        self.B = LipschitzInnerLayer(in_features=self.layer_widths[-1], out_features=self.out_features,
                                     **self.factory_kwargs)

        layer = []

        prev = self.in_features

        for out_features in self.layer_widths:
            layer.append(LipschitzInnerLayer(prev, out_features, constant=self.sqrt_2, **self.factory_kwargs))
            prev = out_features

        self.layers = ModuleList(layer)

    def check_weights(self):
        a_weight = self.A.compute_weight()
        assert torch.linalg.matrix_norm(a_weight.T @ a_weight, ord=2) < self.lipschitz_constant

    def forward(self, input: Tensor) -> Tensor:
        first, a_weight = self.A(input)

        second = input

        ds = [self.A.compute_d1_sqrt()]

        total_sum = self.A.compute_d1_inv()

        current_product = total_sum

        for i, layer in enumerate(self.layers):
            layer: LipschitzInnerLayer

            second, cn = layer(second, ds[-1])

            ds.append(layer.compute_d_sqrt())

            di_inv = layer.compute_d_inv()

            current_product = current_product @ cn.T

            total_sum = total_sum + (current_product @ di_inv @ current_product.T)

            current_product = current_product @ di_inv

            if isinstance(self.activation, Sequence):
                second = self.activation[i](second)
            else:
                second = self.activation(second)

        smallest_singular_value_cn = torch.linalg.matrix_norm(self.layers[-1].weight, ord=-2)

        B_factor = torch.sqrt(2.0 / (1 + (smallest_singular_value_cn * smallest_singular_value_cn)))

        a_sigma_a_t = B_factor * matrix_inv_square_root(self.B.identity_out + a_weight @ total_sum @ a_weight.T)

        second, b_weight = self.B(second)

        return F.linear(second, a_sigma_a_t, first + self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, inner_widths=({','.join(map(str, self.layer_widths))})"


def train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.reshape((-1 , 1))
        labels = labels.reshape((-1 , 1))

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(x, y, model):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('../runs/sin_trainer_{}'.format(timestamp))

    writer.add_graph(model, torch.randn((10, model.in_features), device=model.device))

    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(x, y)

    training_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(training_loader, optimizer, model, loss_fn, epoch_number, writer)

        print('LOSS train {}'.format(avg_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training',
                           {'Training': avg_loss},
                           epoch_number + 1)
        writer.flush()

        epoch_number += 1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on device", device)

    input_features = 1
    output_features = 1

    model = DeepLipschitzResNet(input_features, output_features, (32, 128, 128, 32), device=device)

    print(model)

    torch.manual_seed(0)

    batch = 64

    x = torch.linspace(-10, 10, 1000, device=device)

    y = torch.sin(x + 1) * 0.5

    train(x, y, model)


if __name__ == '__main__':
    main()
