from typing import Optional, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt

from models.layers import LipschitzLinearLayer
from models.model import DeepLipschitzLinearResNet
from utils import matrix_inv_square_root

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
        result = torch.cat((torch.tensor([0],device=device), torch.cumsum(torch.tensor(dimensions, device=device), dim=0)))

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

        CC = torch.zeros((Ci.shape[0] * 2, Ci.shape[1]  + Ci.shape[0]), device=device)
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

def extract_lmi_constants(model : DeepLipschitzLinearResNet):
    a_weight = model.A.compute_weight()

    prev_d = model.A.compute_d1_sqrt()

    total_sum = model.A.compute_d1_inv()

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
            prev_d = layer.compute_d_sqrt()

        di_inv = layer.compute_d_inv()

        current_product = current_product @ cn.T

        total_sum = total_sum + (current_product @ di_inv @ current_product.T)

        current_product = current_product @ di_inv

    smallest_singular_value_cn = torch.linalg.matrix_norm(model.layers[-1].weight, ord=-2)

    b_factor = torch.sqrt(2.0 / (1 + (smallest_singular_value_cn * smallest_singular_value_cn)))

    a_sigma_a_t = b_factor * matrix_inv_square_root(model.B.identity_out + a_weight @ total_sum @ a_weight.T)

    b_weight = a_sigma_a_t @ model.B.compute_weight()

    return a_weight, b_weight, Cs, Ls, Lms, dimensions

def plot_matrix(M: torch.Tensor, fig_name:str, dpi=300):
    min, max = M.min(), M.max()

    normalized = (M - min) / (max - min) * 255
    normalized = normalized.round().to(torch.uint8)

    normalized = normalized.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(normalized, cmap='gray')
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=dpi)

def freedman_diaconis_bins(x: torch.Tensor):
    x = x.flatten()
    n = x.numel()
    q25, q75 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    iqr = q75 - q25
    h = 2 * iqr / (n ** (1/3))
    h = h.item()
    if h == 0:
        return int(np.sqrt(n))  # fallback: sqrt rule
    bins = int(torch.ceil((x.max() - x.min()) / h).item())
    return max(1, bins)

def plot_histogram(x: torch.Tensor, fig_name:str, dpi=300):
    fig, ax = plt.subplots()
    counts, edges = torch.histogram(x.cpu(), bins=50, density=True)

    ax.bar(edges[:-1].numpy(), counts.numpy(),
           width=(edges[1] - edges[0]).item())
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Torch Vector")
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png", dpi=dpi)

if __name__ == '__main__':
    torch.manual_seed(0)

    input_features = 40
    output_features = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.no_grad()

    model = DeepLipschitzLinearResNet(input_features, output_features, (32, 256, 1024, 1024), device=device)

    with torch.no_grad():
        a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

    M = main_real(a_weight, b_weight, Cs, Ls, Lms, dimensions, device=device)

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    print(M)

    assert torch.all(eigenvalues < 0), "Not all the eigenvalues of the LMI are negative"

    print(eigenvalues)

    plot_matrix(M, "full_lmi")

    removed_diag = (M - torch.diag(torch.diag(M)))

    plot_matrix(removed_diag, "lmi_without_diag")

    binary = (torch.abs(M) > 0).to(torch.uint8)

    plot_matrix(binary, "binary_lmi")

    plot_matrix(eigenvectors, "eigenvectors_lmi")

    plot_histogram(eigenvalues, "eigenvalues_lmi")


