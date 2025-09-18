import torch

# Code for issue: https://github.com/pytorch/pytorch/issues/163243

def repro():
    n = 8

    dtype = torch.complex64


    A = torch.randn(4, n, n, dtype=dtype, requires_grad=True)
    A = A.clone(memory_format=torch.contiguous_format)

    I0 = torch.eye(n, dtype=A.dtype, device=A.device)  # (n, n)
    I = I0.unsqueeze(0).expand(A.shape[0], n, n).contiguous()

    A = I + 0.5 * (A @ A.mH)  # make Hermitian

    R = torch.linalg.cholesky(A, upper=True)
    loss = R.abs().sum()
    loss.backward()


if __name__ == '__main__':
    repro = torch.compile(repro, backend="aot_eager_decomp_partition")
    repro()
