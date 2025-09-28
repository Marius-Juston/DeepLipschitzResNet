import logging
import traceback

import torch


def safe_cholesky(A: torch.Tensor, upper=False, eps=1e-4):
    res, info = torch.linalg.cholesky_ex(A, upper=upper)

    if info.any():
        A64 = A.to(torch.float64)

        A_sym = (A64 + A64.mT) / 2
        w, V = torch.linalg.eigh(A_sym)

        w_clamped = torch.clamp(w, min=eps)

        D = torch.diag_embed(w_clamped)

        B_pos = V.matmul(D).matmul(V.mT)

        res, info = torch.linalg.cholesky_ex(B_pos, upper=upper)

        if info.any():
            logging.error(f"Chokesy failed: {info}")

            raise ValueError(f"Unable to perform Cholesky.\n{'\n'.join(traceback.format_stack())}")

        return res.to(A.dtype)

    return res


def safe_cholesky_cond(A: torch.Tensor, upper: bool = False, min_eps: float = 1e-4, max_eps: float = 1e5) -> torch.Tensor:
    res, info = torch.linalg.cholesky_ex(A, upper=upper)

    def true_fn(A, res, upper):
        A64 = A.to(torch.float64)

        A_sym = (A64 + A64.mT) / 2.0
        w, V = torch.linalg.eigh(A_sym)
        w_pos = torch.clamp(w, min=min_eps, max=max_eps)
        D = torch.diag_embed(w_pos)
        B_pos = V.matmul(D).matmul(V.mT)
        res2, _ = torch.linalg.cholesky_ex(B_pos, upper=upper)
        return res2.to(A.dtype)

    def false_fn(A, res, upper):
        return res.clone()

    return torch.cond(info.any(), true_fn, false_fn, (A, res, upper))


if __name__ == '__main__':
    A = torch.randn((3, 3))

    # safe_cholesky = torch.compile(safe_cholesky, fullgraph=True)
    #
    # try:
    #     safe_cholesky(A)
    # except:
    #     traceback.print_exc()

    res = torch._dynamo.explain(safe_cholesky_cond)(A)

    print(res)
