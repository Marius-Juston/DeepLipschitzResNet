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


def safe_cholesky(A: torch.Tensor, upper: bool = False, min_eps: float = 1e-4, max_eps: float = 1e5) -> torch.Tensor:
    was_batched = (A.ndim == 3)
    if not was_batched:
        A = A.unsqueeze(0)

    A = A.contiguous()
    res, info = torch.linalg.cholesky_ex(A, upper=upper)
    res = res.contiguous()

    def true_fn(A_operand, res_operand, min_eps_operand, max_eps_operand):
        A64 = A_operand.to(torch.float64)
        A_sym = (A64 + A64.mT) / 2.0
        w, V = torch.linalg.eigh(A_sym)
        w_pos = torch.clamp(w, min=min_eps_operand, max=max_eps_operand)
        D = torch.diag_embed(w_pos)
        B_pos = V.matmul(D).matmul(V.mT)
        res2, _ = torch.linalg.cholesky_ex(B_pos, upper=upper)
        return res2.to(A_operand.dtype).contiguous().clone().contiguous()

    def false_fn(A_operand, res_operand, min_eps_operand, max_eps_operand):
        return res_operand.contiguous().clone().contiguous()

    min_eps_tensor = torch.tensor(min_eps, dtype=A.dtype, device=A.device)
    max_eps_tensor = torch.tensor(max_eps, dtype=A.dtype, device=A.device)

    operands = (A, res, min_eps_tensor, max_eps_tensor)
    out = torch.cond(info.any(), true_fn, false_fn, operands)

    if not was_batched:
        out = out.squeeze(0)

    return out


if __name__ == '__main__':
    A = torch.randn((3, 3), requires_grad=True)

    A = A @ A.T

    B = torch.randn((10, 3, 3), requires_grad=True)

    safe_cholesky = torch.compile(safe_cholesky_cond, mode='max-autotune', fullgraph=True)

    G = safe_cholesky(A)
    print(G)
    G = safe_cholesky(B)

    print(G)

    res = torch._dynamo.explain(safe_cholesky_cond)(A)

    print(res)
