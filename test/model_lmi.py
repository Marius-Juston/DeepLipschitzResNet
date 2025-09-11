import os.path
from typing import Optional, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, ConnectionPatch

import utils
from models.layers import LipschitzLinearLayer
from models.model import DeepLipschitzLinearResNet


def block(tensors: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    return torch.cat([torch.cat(b, dim=1) for b in tensors], dim=0)


def E_real(dimensions: Sequence[int], i: Optional[int], device=None):
    Dn = sum(dimensions)

    if i is None:
        d1, d2 = dimensions[0], dimensions[-1]
    else:
        d1, d2 = dimensions[i], dimensions[i + 1]

    E = torch.zeros((Dn, d1 + d2), device=device)
    id_1 = torch.eye(d1, device=device)
    id_2 = torch.eye(d2, device=device)

    if i is None:
        E[:d1, :d1] = id_1
        E[-d2:, -d2:] = id_2
    else:
        result = torch.cat(
            (torch.tensor([0], device=device), torch.cumsum(torch.tensor(dimensions, device=device), dim=0)))

        E[result[i]: result[i + 1], :d1] = id_1
        E[result[i + 1]: result[i + 2], -d2:] = id_2

    return E.T


def main_real(A, B, Cs, Ls, Lms, dimensions, device=None):
    L = len(Ls)

    E0 = E_real(dimensions, None, device=device)

    I = torch.eye(dimensions[0], device=device)

    M0 = block([[A.T @ A - I, A.T @ B], [B.T @ A, B.T @ B]])

    M0 = E0.T @ M0 @ E0

    Ms = [M0]

    for i in range(L):
        Ei = E_real(dimensions, i, device=device)

        lambda_i = Ls[i]
        Ci = Cs[i]

        CC = torch.zeros((Ci.shape[0] * 2, Ci.shape[1] + Ci.shape[0]), device=device)
        CC[:Ci.shape[0], :Ci.shape[1]] = Ci
        CC[-Ci.shape[0]:, -Ci.shape[0]:] = torch.eye(Ci.shape[0], device=device)

        L_, m = Lms[i]

        M_sub = block(
            [
                [-2 * L_ * m * lambda_i, (m + L_) * lambda_i],
                [(m + L_) * lambda_i, -2 * lambda_i],
            ]
        )

        Mi = CC.T @ M_sub @ CC
        expanded_Mi = Ei.T @ Mi @ Ei

        Ms.append(expanded_Mi)

    output_M = Ms[0]

    for i in range(1, len(Ms)):
        output_M += Ms[i]

    # eigs = np.linalg.eigh(output_M)

    # print(eigs)

    M = output_M

    return M


def extract_lmi_constants(model: DeepLipschitzLinearResNet):
    a_weight = model.A.compute_weight()

    assert torch.linalg.matrix_norm(a_weight, ord=2) <= model.lipschitz_constant

    prev_d = model.A.compute_d_extra(transpose=True)

    total_sum = model.A.compute_d_inv(transpose=True)

    current_product = total_sum

    Cs = []
    Ls = []
    Lms = []
    dimensions = [
        a_weight.shape[1],
    ]

    for i, layer in enumerate(model.layers):
        layer: LipschitzLinearLayer

        cn = layer.compute_weight(prev_d)
        Cs.append(cn)
        Ls.append(layer.identity_out)
        Lms.append([1, 0])
        dimensions.append(cn.shape[0])

        if i < len(model.layers) - 1:
            prev_d = layer.compute_d_extra()

        di_inv = layer.compute_d_inv()

        current_product = current_product @ cn.T

        total_sum = total_sum + (current_product @ di_inv @ current_product.T)

        current_product = current_product @ di_inv

    a_sigma_a_t, b_factor = model.compute_b_factor(a_weight, total_sum)
    b_weight = model.B.compute_weight(a_sigma_a_t, left=True)

    return a_weight, b_weight, Cs, Ls, Lms, dimensions


def plot_matrix(M: torch.Tensor, fig_name: str= None, dpi:int=300, mask:bool=True, ax:Optional[Axes]=None):
    if mask:
        masked_zerp = M == 0
        no_mask = M != 0

        min, max = M[no_mask].min(), M[no_mask].max()
    else:
        min, max = M.min(), M.max()

    normalized = (M - min) / (max - min) * 255
    normalized = normalized.round()

    if mask:
        normalized[masked_zerp] = torch.nan

    normalized = normalized.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(normalized, cmap='viridis', vmin=0, vmax=255)
        fig.tight_layout()
        fig.savefig(f"{fig_name}.png", dpi=dpi)
    else:
        ax.matshow(normalized, cmap='viridis', vmin=0, vmax=255)


def freedman_diaconis_bins(x: torch.Tensor):
    x = x.flatten()
    n = x.numel()
    q25, q75 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    iqr = q75 - q25
    h = 2 * iqr / (n ** (1 / 3))
    h = h.item()
    if h == 0:
        return int(np.sqrt(n))  # fallback: sqrt rule
    bins = int(torch.ceil((x.max() - x.min()) / h).item())
    return max(1, bins)


def plot_histogram(x: torch.Tensor, fig_name: str, dpi=300):
    fig, ax = plt.subplots()
    counts, edges = torch.histogram(x.cpu(), bins=50, density=True)

    ax.bar(edges[:-1].numpy(), counts.numpy(),
           width=(edges[1] - edges[0]).item())
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Torch Vector")
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=dpi)


def singular_extremes(M: torch.Tensor):
    """Return (sigma_min, sigma_max) for matrix M."""
    s = torch.linalg.svdvals(M)
    return s.min().item(), s.max().item()


def plot_singular_extremes(a_weight, b_weight, Cs, fig_name: str, dpi=300):
    matrices = [a_weight] + Cs + [b_weight]
    labels = ["A"] + [f"C{i + 1}" for i in range(len(Cs))] + ["B"]

    sigmas_min, sigmas_max = [], []

    for M in matrices:
        smin, smax = singular_extremes(M)
        sigmas_min.append(smin)
        sigmas_max.append(smax)
    sigma_diff = [a - b for a, b in zip(sigmas_max, sigmas_min)]

    # Plot
    x = range(len(matrices))

    fig, ax = plt.subplots()
    ax.plot(x, sigmas_min, "o-", label="σ_min", markersize=8)
    ax.plot(x, sigmas_max, "s-", label="σ_max", markersize=8)
    ax.plot(x, sigma_diff, "^-", label="σ_diff", markersize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Matrix")
    ax.set_ylabel("Singular value")
    ax.set_title("Extremal Singular Values of Matrices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=dpi)


def test_sqrt(device, tol=5e-3):
    n = 1000

    A = torch.randn(n, n, device=device)

    W = torch.eye(n, device=device) + A @ A.T

    w_gt = torch.linalg.inv(W)

    w_inv_actual = utils.fast_symmetric_positive_definitive_matrix_inverse(W)
    assert torch.dist(w_gt, w_inv_actual) < tol

    # Symmetric version
    L = utils.matrix_inv_square_root(W)
    w_inv = L @ L.T
    assert torch.dist(w_gt, w_inv) < tol
    w_inv = L.T @ L
    assert torch.dist(w_gt, w_inv) < tol

    # Triangular non-symmetric version
    L = utils.matrix_inv_square_root_triangle(W)
    w_inv_triangle = L.T @ L
    assert torch.dist(w_gt, w_inv_triangle) < tol


def plot_zoomed_in_system(A, B, fig_name: str, dpi=300):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    plot_matrix(A, ax=ax1)
    plot_matrix(B, ax=ax2)

    range_lim = (A.shape[0]-B.shape[0], A.shape[0])

    a_size = np.array(A.shape)
    b_size = np.array(B.shape)

    size = b_size
    pos = a_size - b_size

    rect = Rectangle(pos, *size,
                     linewidth=2, edgecolor="black", facecolor="none")
    ax1.add_patch(rect)

    rect = Rectangle((0, 0), *a_size,
                     linewidth=2, edgecolor="black", facecolor="none")
    ax2.add_patch(rect)

    con1 = ConnectionPatch(xyA=pos, coordsA=ax1.transData,
                           xyB=(0, 0), coordsB=ax2.transData,
                           color="black")

    con2 = ConnectionPatch(xyA=pos + size, coordsA=ax1.transData,
                           xyB=b_size[::-1], coordsB=ax2.transData,
                           color="black")

    fig.add_artist(con1)
    fig.add_artist(con2)

    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=dpi)


if __name__ == '__main__':
    torch.manual_seed(0)

    figs_folder = 'figs'
    os.makedirs(figs_folder, exist_ok=True)

    input_features = 4
    output_features = input_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_sqrt(device)

    torch.no_grad()

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 256, 1024, 1024), device=device)

    # save_folder = '/home/mjuston2/Documents/DeepLipschitzResNet/runs/sin_trainer_20250828_155950'
    save_folder = None

    if save_folder is not None:
        data = torch.load(f"{save_folder}/best.pt", map_location=device)
        new_state_dict = {key.replace('_orig_mod.', ''): val for key, val in data.items()}

        model.load_state_dict(new_state_dict)

    with torch.no_grad():
        a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

    M = main_real(a_weight, b_weight, Cs, Ls, Lms, dimensions, device=device)

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    print(M)

    print(eigenvalues)

    plot_matrix(M, os.path.join(figs_folder, "full_lmi"))

    removed_diag = (M - torch.diag(torch.diag(M)))

    plot_matrix(removed_diag, os.path.join(figs_folder, "lmi_without_diag"))

    binary = (torch.abs(M) > 1e-5).to(torch.uint8)

    plot_matrix(binary, os.path.join(figs_folder, "binary_lmi"), mask=False)

    plot_matrix(eigenvectors, os.path.join(figs_folder, "eigenvectors_lmi"))

    B_zoom = b_weight.T @ b_weight
    plot_matrix(B_zoom, os.path.join(figs_folder, "zoomed_in_b"))

    plot_zoomed_in_system(removed_diag, B_zoom, os.path.join(figs_folder, "inplace_zoomed_in_b"))

    plot_histogram(eigenvalues, os.path.join(figs_folder, "eigenvalues_lmi"))

    plot_singular_extremes(a_weight, b_weight, Cs, os.path.join(figs_folder, "singular_weights"))

    assert torch.all(eigenvalues < 1e-3), "Not all the eigenvalues of the LMI are negative"
