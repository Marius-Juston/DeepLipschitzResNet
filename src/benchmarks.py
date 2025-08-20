import torch


# --------------------------
# Implementations
# --------------------------

def invsqrt_eigh(A):
    return torch.linalg.inv(A)

def fast_symmetric_positive_definitive_matrix_inverse(A):
    L = torch.linalg.cholesky_ex(A).L
    return torch.cholesky_inverse(L)


# --------------------------
# Benchmarking helper
# --------------------------

def benchmark(fn, *args, repeat=5):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start.record()
        result = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds
    return result, sum(times) / len(times)




if __name__ == '__main__':

    # --------------------------
    # Run benchmarks
    # --------------------------

    sizes = [512, 2048, 8192, 2 ** 14, 2 ** 15]  # adjust depending on your GPU memory
    dtype = torch.float32
    device = "cuda"

    for n in sizes:
        print(f"\nMatrix size: {n}x{n}")
        # Generate random SPD matrix: A = M M^T + n*I
        M = torch.randn(n, n, device=device, dtype=dtype)
        A = M @ M.T + torch.eye(n, device=device, dtype=dtype)

        # 1. Eigen decomposition
        res_eigh, t_eigh = benchmark(invsqrt_eigh, A)
        print(f"Eigen-decomposition full A^(-1): {t_eigh:.2f} ms")

        res_eigh_G,_ = torch.linalg.eigvalsh(res_eigh).sort()

        # 2. Cholesky full
        res_chol, t_chol = benchmark(fast_symmetric_positive_definitive_matrix_inverse, A)
        print(f"Cholesky + triangular solve full A^(-1): {t_chol:.2f} ms")

        res_chol_G,_ = torch.linalg.eigvalsh(res_chol).sort()

        dist = torch.dist(res_eigh, res_chol)

        assert torch.dist(res_eigh, res_chol) < 1e-2, f"The matrix results are not the same, dist: {dist}"
