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

    return loss


if __name__ == '__main__':
    import torch.utils.benchmark as benchmark

    t0 = benchmark.Timer(
        stmt='repro()',
        setup='from __main__ import repro')

    print(t0.timeit(1000))

    t0 = benchmark.Timer(
        stmt='repro()',
        setup='from __main__ import repro; repro = torch.compile(repro, backend="inductor", mode="max-autotune"); [repro() for i in range(10)]')

    print(t0.timeit(1000))

    t0 = benchmark.Timer(
        stmt='repro()',
        setup='from __main__ import repro; repro = torch.compile(repro, backend="eager"); [repro() for i in range(10)]')

    print(t0.timeit(1000))
