import torch
from torch import Tensor, cholesky_inverse
from torch.linalg import eigh


# @torch.compile
def matrix_inv_square_root(x: Tensor) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    eigvals, eigvecs = eigh(x.detach())
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def matrix_inv_square_root_triangle(x: Tensor, B: Tensor = None, upper: bool = False, left: bool = True) -> Tensor:
    """
    Returns the Matrix's square root inverse x**(-1/2), detaches matrix so that gradients do not cause problems close to 0.
    THIS IS NOT THE "UNIQUE" SYMMETRIC MATRIX INV SQUARE ROOT!!!
    :param B:
    :param upper:
    :param left:
    :param x: Matrix to square root inverse
    :return: The inverse square root of x
    """
    L = torch.linalg.cholesky_ex(x, upper=upper).L

    if B is None:
        B = torch.eye(L.shape[0], dtype=x.dtype, device=x.device)
    L_inv = torch.linalg.solve_triangular(L, B, upper=upper, left=left)
    return L_inv


def symmetrize_(x: Tensor) -> Tensor:
    return (x + x.T) / 2.0


# @torch.compile
def fast_symmetric_positive_definitive_matrix_inverse(x: Tensor) -> Tensor:
    """
    With the assumption that x is a symmetric positive definite (PSD) matrix, take the inverse of the matrix
    :param x: The matrix to invert
    :return: The inverse of the matrix
    """
    L = torch.linalg.cholesky(x)

    return cholesky_inverse(L)


def get_largest_singular_value_power_iteration(matrix: Tensor, num_iterations: int = 25) -> Tensor:
    """
    Computes the largest singular value of a matrix using the Power Iteration method.
    This is a classic and reliable iterative method.

    Args:
        matrix (torch.Tensor): The input matrix.
        num_iterations (int): The number of iterations to run.

    Returns:
        float: The largest singular value.
    """
    v = torch.randn(matrix.shape[1], 1, device=matrix.device)
    ata = matrix.T @ matrix
    for _ in range(num_iterations):
        v_new = ata @ v
        v = v_new / torch.norm(v_new)
    largest_eigenvalue = (v.T @ ata @ v) / (v.T @ v)
    return torch.sqrt(largest_eigenvalue.maximum(torch.tensor(0.0, device=matrix.device)))
