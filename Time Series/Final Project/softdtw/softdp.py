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
# Fast path for soft-DTW_omega: the same arithmetic as soft_dp on the edge costs of
# edge_costs(kind='adtw'), without building the cost and usage arrays. Results are
# bit-identical to the generic engine (tests/test_softdp.py checks it).
# ---------------------------------------------------------------------------

@njit(cache=True)
def adtw_value_and_usage(D_, omega, gamma, R, e, W):
    """Forward and backward soft DP of soft-DTW_omega, in place.

    D_: zero-padded cost (n + 1, m + 1), D_[i, j] = delta_ij for i, j >= 1.
    R, e: work buffers (n + 1, m + 1); W (n, m) receives the expected usage of each
    cell (Pd + Pv + Ph, interior). Returns r[n, m]. Requires gamma > 0.
    """
    n1, m1 = D_.shape
    R[0, :] = np.inf
    R[:, 0] = np.inf
    R[0, 0] = 0.0
    for i in range(1, n1):
        for j in range(1, m1):
            c = D_[i, j]
            cw = c + omega
            R[i, j] = _softmin3(R[i - 1, j - 1] + c, R[i - 1, j] + cw, R[i, j - 1] + cw, gamma)
    # backward in the order of soft_dp_backward (border cells only feed border cells)
    e[:, :] = 0.0
    e[n1 - 1, m1 - 1] = 1.0
    for i in range(n1 - 1, 0, -1):
        for j in range(m1 - 1, 0, -1):
            eij = e[i, j]
            if eij == 0.0:
                W[i - 1, j - 1] = 0.0
                continue
            c = D_[i, j]
            cw = c + omega
            pd = eij * _edge_weight(R[i, j], R[i - 1, j - 1], c, gamma)
            e[i - 1, j - 1] += pd
            pv = eij * _edge_weight(R[i, j], R[i - 1, j], cw, gamma)
            e[i - 1, j] += pv
            ph = eij * _edge_weight(R[i, j], R[i, j - 1], cw, gamma)
            e[i, j - 1] += ph
            W[i - 1, j - 1] = pd + pv + ph
    return R[n1 - 1, m1 - 1]


@njit(cache=True)
def adtw_forward_value(x, c, gamma, omega, R):
    """soft-DTW_omega(x, c) with the cost (x_i - c_j)^2 summed over features, as in
    omega.pairwise_adtw; R is a work buffer of shape (n + 1, m + 1)."""
    n, m, p = x.shape[0], c.shape[0], x.shape[1]
    R[0, :] = np.inf
    R[:, 0] = np.inf
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d2 = 0.0
            for k in range(p):
                diff = x[i - 1, k] - c[j - 1, k]
                d2 += diff ** 2
            cw = d2 + omega
            R[i, j] = _softmin3(R[i - 1, j - 1] + d2, R[i - 1, j] + cw, R[i, j - 1] + cw, gamma)
    return R[n, m]


_workspace = {}


def _buffers(n, m):
    """Work buffers reused across calls of the same shape (one shape per process)."""
    if (n, m) not in _workspace:
        _workspace.clear()
        _workspace[(n, m)] = (np.zeros((n + 1, m + 1)), np.empty((n + 1, m + 1)),
                              np.empty((n + 1, m + 1)), np.empty((n, m)), np.empty((n, m)))
    return _workspace[(n, m)]


def adtw_value_and_grad(x, y, gamma, omega):
    """soft-DTW_omega(x, y) and its gradient wrt x; same result as
    soft_distance_value_and_grad(x, y, gamma, 'adtw', omega=omega), faster."""
    if gamma <= 0:
        raise ValueError("the backward pass needs gamma > 0 (use soft_dp_forward for gamma = 0)")
    x, y = _as_2d(x), _as_2d(y)
    n, m = x.shape[0], y.shape[0]
    D_, R, e, W, M = _buffers(n, m)
    D = D_[1:, 1:]  # sqeuclidean_cost, written in place
    np.add((x ** 2).sum(1)[:, None], (y ** 2).sum(1)[None, :], out=D)
    np.matmul(2.0 * x, y.T, out=M)
    np.subtract(D, M, out=D)
    np.maximum(D, 0.0, out=D)
    value = adtw_value_and_usage(D_, float(omega), float(gamma), R, e, W)
    return value, 2.0 * (W.sum(1)[:, None] * x - W @ y)


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
    if kind == "adtw":  # bit-identical fast path
        return adtw_value_and_grad(x, y, gamma, params.get("omega", 0.0))
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    value, P = soft_dp(Cd, Cv, Ch, gamma)
    return value, _grad_x(x, y, P)


def expected_warps(x, y, gamma, omega=0.0):
    """d soft-DTW_omega / d omega: expected number of down + right (warping) steps."""
    Cd, Cv, Ch = edge_costs(x, y, "adtw", omega=omega)
    _, (Pd, Pv, Ph) = soft_dp(Cd, Cv, Ch, gamma)
    return Pv.sum() + Ph.sum()
