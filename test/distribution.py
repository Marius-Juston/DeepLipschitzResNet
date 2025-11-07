import numpy as np
import torch


@torch.no_grad()
def weight(b, n, std_, device=None, dtype=None):
    w = torch.randn((b, n, n), device=device, dtype=dtype) * std_

    identity = torch.eye(n, device=device, dtype=dtype)
    omega = identity + w @ w.mT

    R = torch.linalg.cholesky(omega, upper=True)

    W = torch.linalg.solve_triangular(R, w, upper=True, left=False)

    return W


if __name__ == '__main__':
    max_b = 100_000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    stds = []

    max_n = 614100000

    for std_ in [0.25, 0.5, 1, 2, 4]:
        for i in range(1, 300):
            b = max_n // (i ** 2)

            W = weight(b, i, std_, device=device, dtype=dtype)

            mean, std = torch.mean(W), torch.std(W)
            torch.cuda.empty_cache()

            stds.append([i, std.item(), mean.item()])

            print(i, b, b * i ** 2, mean.item(), std.item())

            np.savetxt(f'stds_{std_}.csv', stds, delimiter=',')
