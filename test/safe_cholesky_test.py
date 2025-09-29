import torch

from utils import safe_cholesky

if __name__ == '__main__':
    # single SPD matrix
    m = 5
    A = torch.randn(m, m, requires_grad=True)
    A = A @ A.T  # symmetric positive semidefinite
    # intentionally destroy definiteness slightly:
    A[0, 0] -= 1e6
    print(A)
    L = safe_cholesky(A)
    # print(L)

    print(A)

    # verify
    recon = L @ L.T
    torch.allclose(recon, 0.5 * (A + A.T), atol=1e-6)  # approx equal after eigen clamping

    # batched example
    batch = 4
    X = torch.randn(batch, m, m)
    B = X @ X.transpose(-2, -1)
    # make one matrix slightly singular
    B[2] = B[2] - 1e-9 * torch.eye(m)
    Lbat = safe_cholesky(B)  # returns (batch, m, m)
    # print(Lbat)
