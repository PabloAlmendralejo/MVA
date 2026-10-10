"""Soft-DTW in pure PyTorch, vectorised over anti-diagonals (runs on CPU or CUDA).

All cells with the same i + j depend only on the previous two anti-diagonals, so
each anti-diagonal (and the whole batch) is computed in one tensor operation:
n + m - 1 sequential steps instead of n * m. This is the scheme used by GPU
implementations of soft-DTW. Forward and backward follow Algorithms 1 and 2.
Inputs: x (B, n, p), y (B, m, p).
"""

import torch


def _diag_cells(d, n, m, device):
    """1-based (i, j) of the cells with i + j = d inside the n x m grid."""
    i = torch.arange(max(1, d - m), min(n, d - 1) + 1, device=device)
    return i, d - i


def sdtw_forward_diag(D, gamma):
    """D: (B, n, m) costs -> R: (B, n + 2, m + 2), R[:, n, m] = dtw_gamma."""
    B, n, m = D.shape
    R = torch.full((B, n + 2, m + 2), float("inf"), dtype=D.dtype, device=D.device)
    R[:, 0, 0] = 0.0
    for d in range(2, n + m + 1):
        i, j = _diag_cells(d, n, m, D.device)
        prev = torch.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]])  # (3, B, k)
        R[:, i, j] = D[:, i - 1, j - 1] - gamma * torch.logsumexp(-prev / gamma, dim=0)
    return R


def sdtw_backward_diag(D, R, gamma):
    """Algorithm 2 over anti-diagonals. Returns E: (B, n, m)."""
    B, n, m = D.shape
    D_ = torch.zeros((B, n + 2, m + 2), dtype=D.dtype, device=D.device)
    D_[:, 1:n + 1, 1:m + 1] = D
    R_ = R.clone()
    R_[:, :, m + 1] = -float("inf")
    R_[:, n + 1, :] = -float("inf")
    R_[:, n + 1, m + 1] = R[:, n, m]
    E = torch.zeros((B, n + 2, m + 2), dtype=D.dtype, device=D.device)
    E[:, n + 1, m + 1] = 1.0
    for d in range(n + m, 1, -1):
        i, j = _diag_cells(d, n, m, D.device)
        r = R_[:, i, j]
        a = torch.exp((R_[:, i + 1, j] - r - D_[:, i + 1, j]) / gamma)
        b = torch.exp((R_[:, i, j + 1] - r - D_[:, i, j + 1]) / gamma)
        c = torch.exp((R_[:, i + 1, j + 1] - r - D_[:, i + 1, j + 1]) / gamma)
        E[:, i, j] = a * E[:, i + 1, j] + b * E[:, i, j + 1] + c * E[:, i + 1, j + 1]
    return E[:, 1:n + 1, 1:m + 1]


class SoftDTWDiagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gamma):
        D = torch.cdist(x, y) ** 2
        R = sdtw_forward_diag(D, gamma)
        ctx.save_for_backward(x, y, D, R)
        ctx.gamma = gamma
        n, m = D.shape[1], D.shape[2]
        return R[:, n, m]

    @staticmethod
    def backward(ctx, grad_output):
        x, y, D, R = ctx.saved_tensors
        E = sdtw_backward_diag(D, R, ctx.gamma)
        g = grad_output[:, None, None]
        grad_x = 2.0 * (E.sum(2, keepdim=True) * x - E @ y) * g
        grad_y = 2.0 * (E.sum(1).unsqueeze(-1) * y - E.transpose(1, 2) @ x) * g
        return grad_x, grad_y, None


class SoftDTWLossDiag(torch.nn.Module):
    """Same interface as torch_sdtw.SoftDTWLoss, but device-agnostic (CPU / CUDA)."""

    def __init__(self, gamma=1.0, reduction="mean"):
        super().__init__()
        self.gamma, self.reduction = gamma, reduction

    def forward(self, x, y):
        if x.dim() == 2:
            x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        v = SoftDTWDiagFunction.apply(x, y, self.gamma)
        return {"mean": v.mean(), "sum": v.sum(), "none": v}[self.reduction]
