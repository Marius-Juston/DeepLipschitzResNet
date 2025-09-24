"""
Run this script where your DeepLipschitzConvResNet and the convolutional layers are defined
and the batched SpectralNorm class (from the previous message) is available.
"""
import random

import torch
import torch.nn as nn
from torch.nn import Flatten

from models.cnn_model import DeepLipschitzConvResNet
from models.linear_model import DeepLipschitzLinearResNet
from linear_train import print_num_parameters, train


def make_synthetic_dataset(n_samples: int, C: int, H: int, W: int, seed: int = 0, device=None):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, C, H, W, device=device) * 0.6
    # normalize to be roughly within unit ball
    norms = X.view(n_samples, -1).norm(dim=1, keepdim=True) + 1e-12
    X = X / norms.view(-1, 1, 1, 1)
    # small random scaling to spread samples inside ball
    scales = (0.2 + 0.8 * torch.rand(n_samples, 1, 1, 1, device=device))
    return X * scales


def target_g1(X):
    n = X.shape[0]
    return X.view(n, -1).norm(dim=1, keepdim=True)


class LipschitzNetwork(nn.Module):
    def __init__(self, in_channels=1, hidden=32, device=None):
        super().__init__()
        self.conv = nn.Sequential(
            DeepLipschitzConvResNet(
                in_channels=in_channels, out_channels=1, layers_widths=[hidden, hidden],
                kernel_size=3, padding=1, device=device, dtype=torch.float32
            ), Flatten()
        )
        self.fc = DeepLipschitzLinearResNet(256, 1, [hidden, hidden])

    def forward(self, x):
        z = self.conv(x)
        return self.fc(z)


@torch.no_grad()
def finite_difference_lipschitz_estimate(model, X, n_pairs=500, device='cpu'):
    model.eval()
    X = X.to(device)
    n = X.shape[0]
    max_ratio = 0.0
    for _ in range(n_pairs):
        i = random.randrange(n)
        j = random.randrange(n)
        x = X[i:i + 1]
        y = X[j:j + 1]
        fx = model(x).detach().cpu().item()
        fy = model(y).detach().cpu().item()
        denom = (x - y).view(1, -1).norm().item()
        if denom < 1e-12:
            continue
        ratio = abs(fx - fy) / denom
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio


def empirical_gradient_max_norm(model, X, device='cpu'):
    model.eval()
    X = X.to(device)
    max_norm = 0.0
    for i in range(X.shape[0]):
        x = X[i:i + 1].clone().detach().requires_grad_(True).to(device)
        out = model(x)
        out_scalar = out.squeeze()
        if out_scalar.ndim > 0:
            out_scalar = out_scalar.sum()
        out_scalar.backward(retain_graph=False)
        g = x.grad.view(-1).detach()
        nrm = g.norm().item()
        if nrm > max_norm:
            max_norm = nrm
        x.grad = None
    return max_norm


def run_experiment(device='cpu'):
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    n_train, n_test = 512, 256
    C, H, W = 1, 16, 16
    X_train = make_synthetic_dataset(n_train, C, H, W, seed=0, device=device)
    X_test = make_synthetic_dataset(n_test, C, H, W, seed=1, device=device)
    y1_train = target_g1(X_train)
    K = 10.0
    y2_train = K * y1_train

    batch = 64
    epochs = 20

    theoretical_lower = 0
    lr = 1e-4
    termination_error = 1e-4

    print(X_train.shape)

    model = LipschitzNetwork(device=device)

    model = torch.compile(model, mode='max-autotune')

    print(model)
    print(print_num_parameters(model))

    # Now train baseline and lipschitz model on g1
    print("Training unconstrained baseline on g1 (1-Lipschitz)...")
    train(X_train, y1_train, model, learning_prefix='general_cnn', batch_size=batch,
          termination_error=termination_error, lr=lr, theoretical_lower=theoretical_lower, logging_frequency=1,
          epochs=epochs)

    print("Finite-diff lipschitz g1:",
          finite_difference_lipschitz_estimate(model, X_test, n_pairs=1000, device=device))
    print("Grad-max lipschitz g1:", empirical_gradient_max_norm(model, X_test, device=device))

    model = LipschitzNetwork(device=device)

    print("Training Lipschitz network on g2 (should struggle if truly 1-Lipschitz)...")
    train(X_train, y2_train, model, learning_prefix='general_cnn_lipschitz', batch_size=batch,
          termination_error=termination_error, lr=lr, theoretical_lower=theoretical_lower, logging_frequency=1,
          epochs=epochs)

    print(f"\nEvaluation on g2 (K={K}):")
    print("Finite-diff lipschitz g2:",
          finite_difference_lipschitz_estimate(model, X_test, n_pairs=1000, device=device))
    print("Grad-max lipschitz g2:", empirical_gradient_max_norm(model, X_test, device=device))


# ---------- run ----------
if __name__ == "__main__":
    summary = run_experiment()
