from torch import Tensor, cholesky_inverse
from torch.linalg import eigh, cholesky_ex


# @torch.compile
def matrix_inv_square_root(x: Tensor) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    eigvals, eigvecs = eigh(x.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(x: Tensor) -> Tensor:
    """
    With the assumption that x is a symmetric positive definite (PSD) matrix, take the inverse of the matrix
    :param x: The matrix to invert
    :return: The inverse of the matrix
    """
    L = cholesky_ex(x).L
    return cholesky_inverse(L)
