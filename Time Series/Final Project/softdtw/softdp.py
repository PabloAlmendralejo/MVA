"""A general soft dynamic program over the alignment grid, and the soft distances
it yields: soft-DTW and soft-DTW with a warping penalty omega (kind 'adtw', the
soft counterpart of amerced DTW).

Grid and edges
--------------
Nodes are (i, j) for 0 <= i <= n, 0 <= j <= m; the source is (0, 0), the sink (n, m).
Each node (i, j) can be entered by three edges, each with its own cost:
    diagonal  (i-1, j-1) -> (i, j)   cost Cd[i, j]   (x_i matched with y_j)
    down      (i-1, j)   -> (i, j)   cost Cv[i, j]   (x_i advances, y stays)
    right     (i, j-1)   -> (i, j)   cost Ch[i, j]   (y_j advances, x stays)
Cost arrays have shape (n + 1, m + 1); row/column 0 hold the border edges
(set them to +inf to forbid them, as DTW does). The soft DP is

    r[0, 0] = 0,   r[i, j] = softmin_gamma( r[i-1, j-1] + Cd[i, j],
                                            r[i-1, j]   + Cv[i, j],
                                            r[i, j-1]   + Ch[i, j] )

so r[n, m] = softmin over all source-sink paths of their total cost
(Mensch & Blondel, 2018). The backward pass returns, for every edge, the
derivative of r[n, m] w.r.t. its cost = the expected number of times the edge is
used under the Gibbs distribution over paths. Soft-DTW is the special case
Cd = Cv = Ch = Delta on the interior with forbidden borders.
"""

import numpy as np
from numba import njit

from .core import _as_2d, sqeuclidean_cost


# ---------------------------------------------------------------------------
# Generic engine
# ---------------------------------------------------------------------------

@njit(cache=True)
def _softmin3(a, b, c, gamma):
    mn = min(a, b, c)
    if mn == np.inf:
        return np.inf
    if gamma == 0.0:
        return mn
    s = np.exp(-(a - mn) / gamma) + np.exp(-(b - mn) / gamma) + np.exp(-(c - mn) / gamma)
    return mn - gamma * np.log(s)


@njit(cache=True)
def soft_dp_forward(Cd, Cv, Ch, gamma):
    n1, m1 = Cd.shape
    R = np.full((n1, m1), np.inf)
    R[0, 0] = 0.0
    for i in range(n1):
        for j in range(m1):
            if i == 0 and j == 0:
                continue
            d = R[i - 1, j - 1] + Cd[i, j] if (i > 0 and j > 0) else np.inf
            v = R[i - 1, j] + Cv[i, j] if i > 0 else np.inf
            h = R[i, j - 1] + Ch[i, j] if j > 0 else np.inf
            R[i, j] = _softmin3(d, v, h, gamma)
    return R


@njit(cache=True)
def _edge_weight(r_to, r_from, cost, gamma):
    """d r_to / d (r_from + cost): softmax weight of that edge (argmin indicator if gamma = 0)."""
    if r_to == np.inf or r_from == np.inf or cost == np.inf:
        return 0.0
    if gamma == 0.0:
        return 1.0 if abs(r_from + cost - r_to) < 1e-12 * max(1.0, abs(r_to)) else 0.0
    return np.exp((r_to - r_from - cost) / gamma)


@njit(cache=True)
def soft_dp_backward(Cd, Cv, Ch, R, gamma):
    """Returns (Pd, Pv, Ph): expected usage of every edge = d r[n,m] / d C."""
    n1, m1 = Cd.shape
    e = np.zeros((n1, m1))  # e[i, j] = d r[n, m] / d r[i, j]
    Pd = np.zeros((n1, m1))
    Pv = np.zeros((n1, m1))
    Ph = np.zeros((n1, m1))
    e[n1 - 1, m1 - 1] = 1.0
    for i in range(n1 - 1, -1, -1):
        for j in range(m1 - 1, -1, -1):
            if e[i, j] == 0.0:
                continue
            if i > 0 and j > 0:
                w = _edge_weight(R[i, j], R[i - 1, j - 1], Cd[i, j], gamma)
                Pd[i, j] = e[i, j] * w
                e[i - 1, j - 1] += Pd[i, j]
            if i > 0:
                w = _edge_weight(R[i, j], R[i - 1, j], Cv[i, j], gamma)
                Pv[i, j] = e[i, j] * w
                e[i - 1, j] += Pv[i, j]
            if j > 0:
                w = _edge_weight(R[i, j], R[i, j - 1], Ch[i, j], gamma)
                Ph[i, j] = e[i, j] * w
                e[i, j - 1] += Ph[i, j]
    return Pd, Pv, Ph


def soft_dp(Cd, Cv, Ch, gamma):
    """Value r[n, m] and expected edge usages (Pd, Pv, Ph). Requires gamma > 0."""
    if gamma <= 0:
        raise ValueError("the backward pass needs gamma > 0 (use soft_dp_forward for gamma = 0)")
    R = soft_dp_forward(Cd, Cv, Ch, float(gamma))
    return R[-1, -1], soft_dp_backward(Cd, Cv, Ch, R, float(gamma))


# ---------------------------------------------------------------------------
# Distances: each one is a choice of edge costs + the chain rule back to x
# ---------------------------------------------------------------------------

def _forbid_borders(C):
    C[0, :] = np.inf
    C[:, 0] = np.inf
    return C


def _pad(M):
    n, m = M.shape
    out = np.zeros((n + 1, m + 1))
    out[1:, 1:] = M
    return out


def edge_costs(x, y, kind="dtw", omega=0.0):
    """Edge cost arrays (Cd, Cv, Ch) of shape (n+1, m+1) for a given distance.

    kind = 'dtw'   : all three edges cost delta_ij (soft-DTW)
         = 'adtw'  : warping edges (down/right) cost delta_ij + omega
                     (soft-DTW with a warping penalty)
    """
    D = _pad(sqeuclidean_cost(x, y))
    if kind == "dtw":
        Cd = _forbid_borders(D.copy())
        return Cd, Cd.copy(), Cd.copy()
    if kind == "adtw":
        Cd = _forbid_borders(D.copy())
        Cw = _forbid_borders(D + omega)
        return Cd, Cw, Cw.copy()
    raise ValueError(kind)


def _grad_x(x, y, P):
    """Chain rule from the expected edge usages P = (Pd, Pv, Ph) to x.

    For both kinds every edge entering cell (i, j) costs delta_ij (+ a constant),
    so d/dx_i = sum_j W_ij 2 (x_i - y_j) with W the total expected usage of the cell.
    """
    x, y = _as_2d(x), _as_2d(y)
    W = sum(Q[1:, 1:] for Q in P)
    return 2.0 * (W.sum(1)[:, None] * x - W @ y)


def soft_distance(x, y, gamma, kind="dtw", **params):
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    return soft_dp_forward(Cd, Cv, Ch, float(gamma))[-1, -1]


def soft_distance_value_and_grad(x, y, gamma, kind="dtw", **params):
    """Value and gradient wrt x (see expected_warps for d/d omega)."""
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    value, P = soft_dp(Cd, Cv, Ch, gamma)
    return value, _grad_x(x, y, P)


def expected_warps(x, y, gamma, omega=0.0):
    """d soft-DTW_omega / d omega: expected number of down + right (warping) steps."""
    Cd, Cv, Ch = edge_costs(x, y, "adtw", omega=omega)
    _, (Pd, Pv, Ph) = soft_dp(Cd, Cv, Ch, gamma)
    return Pv.sum() + Ph.sum()
