import torch

from utils import cholesky_rank_k

if __name__ == "__main__":
    # Small gradient check (double precision)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)

    n = 10
    k = 10

    A = torch.randn(n, n)
    A = A @ A.T + torch.eye(n) * 1e-1
    L = torch.linalg.cholesky(A)
    M = torch.randn(n, k)

    L = L.clone().requires_grad_(True)
    M = M.clone().requires_grad_(True)

    # Forward using our op
    # cholesky_rank_k = torch.compile(cholesky_rank_k, mode='max-autotune')

    Lp = cholesky_rank_k(L, M)

    # Compare with naive differentiable path (recompute full cholesky)
    Aprime = (L @ L.T) + (M @ M.T)
    Lp_direct = torch.linalg.cholesky(Aprime)

    # Check forward values are close
    print("max abs forward diff:", (Lp - Lp_direct).abs().max().item())

    # gradcheck (may be slow)
    from torch.autograd import gradcheck

    print("Running gradcheck...")
    ok = gradcheck(lambda a, b: cholesky_rank_k(a, b), (L, M), eps=1e-2, atol=1e-2, rtol=1e-2,
                   check_grad_dtypes=True, fast_mode=False)
    print("gradcheck:", ok)
