"""Soft-DTW as a PyTorch loss, with the hand-written backward of Algorithm 2.

The recursion runs in Numba on CPU (it is inherently sequential, cell by cell);
PyTorch only sees a custom autograd.Function, so soft-DTW can be plugged into
any network. Inputs: x of shape (B, n, p), y of shape (B, m, p).
"""

import numpy as np
import torch
from numba import njit, prange

from softdp import soft_dp_backward, soft_dp_forward
from softdtw import sdtw_forward, sdtw_backward


@njit(parallel=True, cache=True)
def _batch_value_and_alignment(D, gamma, omega):
    """D: (B, n, m) cost matrices -> values (B,), alignments E (B, n, m).

    omega = 0: soft-DTW (Algorithms 1-2). omega > 0: soft-DTW with a warping
    penalty, through the general soft DP of softdp.py; E is then the expected
    usage of each cell (summed over the three edges entering it).
    """
    B, n, m = D.shape
    values = np.empty(B)
    E = np.empty((B, n, m))
    for b in prange(B):
        if omega == 0.0:
            R = sdtw_forward(D[b], gamma)
            values[b] = R[n, m]
            E[b] = sdtw_backward(D[b], R, gamma)
        else:
            Cd = np.full((n + 1, m + 1), np.inf)
            Cd[1:, 1:] = D[b]
            Cw = Cd + omega
            R = soft_dp_forward(Cd, Cw, Cw, gamma)
            values[b] = R[n, m]
            Pd, Pv, Ph = soft_dp_backward(Cd, Cw, Cw, R, gamma)
            E[b] = (Pd + Pv + Ph)[1:, 1:]
    return values, E


class SoftDTWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gamma, omega=0.0):
        D = torch.cdist(x, y) ** 2  # (B, n, m) squared Euclidean costs
        values, E = _batch_value_and_alignment(
            D.detach().cpu().double().numpy(), float(gamma), float(omega))
        E = torch.from_numpy(E).to(dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, y, E)
        return torch.from_numpy(values).to(dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, E = ctx.saved_tensors
        g = grad_output[:, None, None]  # chain rule: one scalar per pair
        # d/dx_i = sum_j E_ij 2 (x_i - y_j) ;  d/dy_j = sum_i E_ij 2 (y_j - x_i)
        grad_x = 2.0 * (E.sum(2, keepdim=True) * x - E @ y) * g
        grad_y = 2.0 * (E.sum(1).unsqueeze(-1) * y - E.transpose(1, 2) @ x) * g
        return grad_x, grad_y, None, None


class SoftDTWLoss(torch.nn.Module):
    """Soft-DTW loss between batches of series (B, n, p) and (B, m, p).

    omega > 0 adds a penalty to every warping step (soft-DTW_omega).
    """

    def __init__(self, gamma=1.0, reduction="mean", omega=0.0):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.gamma, self.omega = gamma, omega
        self.reduction = reduction

    def forward(self, x, y):
        if x.dim() == 2:  # (B, n) univariate -> (B, n, 1)
            x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        values = SoftDTWFunction.apply(x, y, self.gamma, self.omega)
        if self.reduction == "mean":
            return values.mean()
        if self.reduction == "sum":
            return values.sum()
        return values


def soft_dtw_torch_naive(x, y, gamma):
    """Plain-PyTorch forward (cell-by-cell, differentiated by autograd).

    Slow; only used to check the hand-written backward. x: (n, p), y: (m, p).
    """
    D = torch.cdist(x, y) ** 2
    n, m = D.shape
    inf = torch.tensor(float("inf"), dtype=x.dtype)
    R = [[inf] * (m + 1) for _ in range(n + 1)]
    R[0][0] = torch.tensor(0.0, dtype=x.dtype)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]])
            R[i][j] = D[i - 1, j - 1] - gamma * torch.logsumexp(-prev / gamma, 0)
    return R[n][m]
