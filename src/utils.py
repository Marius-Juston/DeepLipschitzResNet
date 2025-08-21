# @torch.compile
from torch.linalg import eigh, cholesky_ex
from torch import Tensor, cholesky_inverse


def matrix_inv_square_root(A: Tensor) -> Tensor:
    eigvals, eigvecs = eigh(A.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(A):
    L = cholesky_ex(A).L
    return cholesky_inverse(L)