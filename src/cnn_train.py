"""
Run this script where your DeepLipschitzConvResNet and the convolutional layers are defined
and the batched SpectralNorm class (from the previous message) is available.
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.cnn_model import DeepLipschitzConvResNet, SpectralNorm

# ---- assume these exist in the environment (from earlier messages) ----
# from your_module import DeepLipschitzConvResNet, SpectralNorm
# If they're in the same file, no import is needed.

# Fallback check
if "DeepLipschitzConvResNet" not in globals():
    raise RuntimeError(
        "DeepLipschitzConvResNet must be defined in the environment (paste the class from previous message).")
if "SpectralNorm" not in globals():
    raise RuntimeError(
        "SpectralNorm must be defined in the environment (use the batched SpectralNorm provided earlier).")


# ---------------- Synthetic dataset helpers ----------------
def make_synthetic_dataset(n_samples: int, C: int, H: int, W: int, seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, C, H, W) * 0.6
    # normalize to be roughly within unit ball
    norms = X.view(n_samples, -1).norm(dim=1, keepdim=True) + 1e-12
    X = X / norms.view(-1, 1, 1, 1)
    # small random scaling to spread samples inside ball
    scales = (0.2 + 0.8 * torch.rand(n_samples, 1, 1, 1))
    return X * scales


def target_g1(X):
    n = X.shape[0]
    return X.view(n, -1).norm(dim=1, keepdim=True)


def target_g2(X, K=2.0):
    return K * target_g1(X, K)


# ---------------- small baseline model ----------------
class SmallConvBaseline(nn.Module):
    def __init__(self, in_channels=1, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.conv(x)
        z = z.view(z.shape[0], -1)
        return self.fc(z)


# ---------------- Evaluation helpers ----------------
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


# ---------------- training loop ----------------
def train_model(model, dataloader, n_epochs=30, lr=1e-3, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = lossfn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
        total_loss /= len(dataloader.dataset)
        if epoch % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch:03d}  loss={total_loss:.6f}")
    return model


# ---------------- experiment runner using DeepLipschitzConvResNet ----------------
def run_experiment(device='cpu'):
    device = torch.device(device)
    # dataset
    n_train, n_test = 512, 256
    C, H, W = 1, 16, 16
    X_train = make_synthetic_dataset(n_train, C, H, W, seed=1)
    X_test = make_synthetic_dataset(n_test, C, H, W, seed=2)
    y1_train = target_g1(X_train)
    y1_test = target_g1(X_test)
    K = 2.0
    y2_train = K * y1_train
    y2_test = K * y1_test

    bs = 64
    loader1 = DataLoader(TensorDataset(X_train, y1_train), batch_size=bs, shuffle=True)
    loader2 = DataLoader(TensorDataset(X_train, y2_train), batch_size=bs, shuffle=True)

    # instantiate unconstrained baselines
    baseline1 = SmallConvBaseline(in_channels=C, hidden=32).to(device)
    baseline2 = SmallConvBaseline(in_channels=C, hidden=32).to(device)

    # instantiate Lipschitz conv networks using the DeepLipschitzConvResNet defined earlier
    # choose a modest width and depth
    lipschitz_model1 = DeepLipschitzConvResNet(
        in_channels=C, out_channels=1, layers_widths=[16, 16],
        kernel_size=3, padding=1, device=device, dtype=torch.float32
    ).to(device)

    lipschitz_model2 = DeepLipschitzConvResNet(
        in_channels=C, out_channels=1, layers_widths=[16, 16],
        kernel_size=3, padding=1, device=device, dtype=torch.float32
    ).to(device)

    # Attach batched SpectralNorm to each Lipschitz model (but we must initialize it after a warm-up forward)
    sn1 = SpectralNorm(n_power_iterations=1, eps=1e-6).to(device)
    sn2 = SpectralNorm(n_power_iterations=1, eps=1e-6).to(device)
    lipschitz_model1.spectral_norm = sn1
    lipschitz_model2.spectral_norm = sn2

    # Warm-up forward to populate internal triangular factors (omega_r) and frequency shapes
    # Use a small mini-batch from training data
    with torch.no_grad():
        xb = X_train[:8].to(device)
        _ = lipschitz_model1(
            xb)  # first forward will run compute_b with spectral_norm present but uninitialized -> that's ok
        # get the prev_r from last inner layer (set by InnerWeightLipschitzConvLayer.forward)
        prev_r = lipschitz_model1.layers[-1].omega_r  # shape (F, m, m) complex or real
        # compute triangular inverse (batched) using model helper
        inner_inverse = lipschitz_model1._triangular_inverse_batch(prev_r, upper=True)
        # initialize spectral norm with this batched matrix
        lipschitz_model1.spectral_norm.initialize(inner_inverse)

        # Repeat for the other model
        _ = lipschitz_model2(xb)
        prev_r2 = lipschitz_model2.layers[-1].omega_r
        inner_inverse2 = lipschitz_model2._triangular_inverse_batch(prev_r2, upper=True)
        lipschitz_model2.spectral_norm.initialize(inner_inverse2)

    # Now train baseline and lipschitz model on g1
    print("Training unconstrained baseline on g1 (1-Lipschitz)...")
    baseline1 = train_model(baseline1, loader1, n_epochs=40, lr=1e-3, device=device)
    print("Training Lipschitz network on g1 (1-Lipschitz)...")
    lipschitz_model1 = train_model(lipschitz_model1, loader1, n_epochs=40, lr=1e-3, device=device)

    # Evaluate on test
    def mse_on_test(model, Xtest, ytest):
        model.eval()
        with torch.no_grad():
            pred = model(Xtest.to(device)).cpu()
            return ((pred - ytest) ** 2).mean().item()

    print("\nEvaluation on g1 (1-Lipschitz):")
    print("MSE baseline g1:", mse_on_test(baseline1, X_test, y1_test))
    print("MSE lipschitz g1:", mse_on_test(lipschitz_model1, X_test, y1_test))
    print("Finite-diff baseline g1:",
          finite_difference_lipschitz_estimate(baseline1, X_test, n_pairs=1000, device=device))
    print("Finite-diff lipschitz g1:",
          finite_difference_lipschitz_estimate(lipschitz_model1, X_test, n_pairs=1000, device=device))
    print("Grad-max baseline g1:", empirical_gradient_max_norm(baseline1, X_test, device=device))
    print("Grad-max lipschitz g1:", empirical_gradient_max_norm(lipschitz_model1, X_test, device=device))

    # Now train on non-1 target g2 (K=2)
    print("\nTraining unconstrained baseline on g2 (K=2)...")
    baseline2 = train_model(baseline2, loader2, n_epochs=40, lr=1e-3, device=device)
    print("Training Lipschitz network on g2 (should struggle if truly 1-Lipschitz)...")
    lipschitz_model2 = train_model(lipschitz_model2, loader2, n_epochs=40, lr=1e-3, device=device)

    print("\nEvaluation on g2 (K=2):")
    print("MSE baseline g2:", mse_on_test(baseline2, X_test, y2_test))
    print("MSE lipschitz g2:", mse_on_test(lipschitz_model2, X_test, y2_test))
    print("Finite-diff baseline g2:",
          finite_difference_lipschitz_estimate(baseline2, X_test, n_pairs=1000, device=device))
    print("Finite-diff lipschitz g2:",
          finite_difference_lipschitz_estimate(lipschitz_model2, X_test, n_pairs=1000, device=device))
    print("Grad-max baseline g2:", empirical_gradient_max_norm(baseline2, X_test, device=device))
    print("Grad-max lipschitz g2:", empirical_gradient_max_norm(lipschitz_model2, X_test, device=device))

    # return a small summary
    return {
        'baseline_g1_mse': mse_on_test(baseline1, X_test, y1_test),
        'lip_g1_mse': mse_on_test(lipschitz_model1, X_test, y1_test),
        'baseline_g2_mse': mse_on_test(baseline2, X_test, y2_test),
        'lip_g2_mse': mse_on_test(lipschitz_model2, X_test, y2_test),
    }


# ---------- run ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = run_experiment(device=device)
    print("Summary:", summary)
