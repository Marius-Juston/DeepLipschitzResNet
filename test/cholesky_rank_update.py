import torch

# https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
# @torch.compile
def rank_1_update(L: torch.Tensor, x: torch.Tensor):
    n = x.shape[0]
    L = L.clone()
    x = x.clone()

    for k in range(n):
        r = torch.hypot(L[k, k], x[k])

        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r

        if k + 1  < n:
            L[k + 1: n, k] = (L[k + 1: n, k] + s * x[k + 1: n]) / c
            x[k + 1: n] = c * x[k + 1: n] - s * L[k + 1: n, k]

    return L

# @torch.compile
def rank_k_update(L: torch.Tensor, M: torch.Tensor):
    for i in range(M.shape[0]):
        L = rank_1_update(L, M[:, i])

    return L


def main():
    torch.manual_seed(0)

    n = 10

    M = torch.randn(n, n)

    A = torch.eye(n) + M @ M.T

    eigenvalues_a = torch.linalg.eigvalsh(A.double()).float()

    L = torch.linalg.cholesky(A)

    M = torch.randn(n, n)
    QM = torch.linalg.qr(M)[0]
    min_eigenvalue = -4
    max_eigenvalue = 1

    eigenvalues = torch.logspace(min_eigenvalue, max_eigenvalue, n).double()

    M = QM.double() @ torch.diag(eigenvalues).double() @ QM.double().T

    computed_eigenvals = torch.linalg.eigvalsh(M)

    print(torch.dist(computed_eigenvals, eigenvalues))

    M = M.float()
    computed_eigenvals = torch.linalg.eigvalsh(M)

    print(torch.dist(computed_eigenvals, eigenvalues))

    new_l = rank_k_update(L, M)

    estimated_val = new_l @ new_l.T
    true_val = A.double() + M.double() @ M.double().T

    print(torch.dist(estimated_val, true_val))

    expected_eigenvalues = eigenvalues_a.double() + eigenvalues.double()**2

    estimated_eigs = torch.linalg.eigvalsh(estimated_val.double())
    print(torch.dist(estimated_eigs, expected_eigenvalues))

    true_eigs = torch.linalg.eigvalsh(true_val.double())
    print(torch.dist(true_eigs, expected_eigenvalues))

    print(estimated_eigs > 0)
    print(true_eigs > 0)


if __name__ == '__main__':
    main()
