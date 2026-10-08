"""A general soft dynamic program over the alignment grid, and the soft elastic
distances it yields (soft-DTW, soft-ADTW, soft-WDTW, soft-ERP) plus their
debiased "divergence" versions.

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

from softdtw import _as_2d, sqeuclidean_cost


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


def wdtw_weights(n, m, g=0.05, wmax=1.0):
    """Logistic weight on the phase difference |i - j| (Jeong et al., 2011)."""
    i, j = np.arange(1, n + 1)[:, None], np.arange(1, m + 1)[None, :]
    half = max(n, m) / 2.0
    return wmax / (1.0 + np.exp(-g * (np.abs(i - j) - half)))


def edge_costs(x, y, kind="dtw", omega=0.0, g=0.05, gap=0.0):
    """Edge cost arrays (Cd, Cv, Ch) of shape (n+1, m+1) for a given distance.

    kind = 'dtw'   : all three edges cost delta_ij (soft-DTW)
         = 'adtw'  : warping edges (down/right) cost delta_ij + omega (amerced DTW)
         = 'wdtw'  : all edges cost w(|i-j|) * delta_ij (weighted DTW)
         = 'erp'   : diagonal costs ||x_i - y_j||^2, down costs ||x_i - gap||^2
                     (x_i aligned to a gap), right costs ||y_j - gap||^2; borders allowed
    """
    x, y = _as_2d(x), _as_2d(y)
    n, m = len(x), len(y)
    D = _pad(sqeuclidean_cost(x, y))
    if kind == "dtw":
        Cd = _forbid_borders(D.copy())
        return Cd, Cd.copy(), Cd.copy()
    if kind == "adtw":
        Cd = _forbid_borders(D.copy())
        Cw = _forbid_borders(D + omega)
        return Cd, Cw, Cw.copy()
    if kind == "wdtw":
        C = _forbid_borders(D * _pad(wdtw_weights(n, m, g)))
        return C, C.copy(), C.copy()
    if kind == "erp":
        gx = ((x - gap) ** 2).sum(1)  # cost of matching x_i with a gap
        gy = ((y - gap) ** 2).sum(1)
        Cd = D.copy()
        Cd[0, :], Cd[:, 0] = np.inf, np.inf
        Cv = np.zeros((n + 1, m + 1))
        Cv[1:, :] = gx[:, None]          # entering (i, j) from (i-1, j), any j >= 0
        Cv[0, :] = np.inf
        Ch = np.zeros((n + 1, m + 1))
        Ch[:, 1:] = gy[None, :]          # entering (i, j) from (i, j-1), any i >= 0
        Ch[:, 0] = np.inf
        return Cd, Cv, Ch
    raise ValueError(kind)


def _grad_x(x, y, P, kind, g=0.05, gap=0.0):
    """Chain rule from the expected edge usages P = (Pd, Pv, Ph) to x."""
    x, y = _as_2d(x), _as_2d(y)
    Pd, Pv, Ph = (Q[1:, 1:] for Q in P)
    if kind in ("dtw", "adtw"):
        W = Pd + Pv + Ph
    elif kind == "wdtw":
        W = (Pd + Pv + Ph) * wdtw_weights(len(x), len(y), g)
    elif kind == "erp":
        # diagonal edges: d||x_i - y_j||^2 ; down edges (all j >= 0): d||x_i - gap||^2
        down = P[1][1:, :].sum(1)
        return 2.0 * (Pd.sum(1)[:, None] * x - Pd @ y) + 2.0 * down[:, None] * (x - gap)
    else:
        raise ValueError(kind)
    return 2.0 * (W.sum(1)[:, None] * x - W @ y)


def soft_distance(x, y, gamma, kind="dtw", **params):
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    return soft_dp_forward(Cd, Cv, Ch, float(gamma))[-1, -1]


def soft_distance_value_and_grad(x, y, gamma, kind="dtw", **params):
    """Value and gradient wrt x. Also returns d/d omega for 'adtw' (= expected #warps)."""
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    value, P = soft_dp(Cd, Cv, Ch, gamma)
    grad = _grad_x(x, y, P, kind, **{k: v for k, v in params.items() if k in ("g", "gap")})
    return value, grad


def expected_warps(x, y, gamma, omega=0.0):
    """d soft-ADTW / d omega: expected number of down + right steps."""
    Cd, Cv, Ch = edge_costs(x, y, "adtw", omega=omega)
    _, (Pd, Pv, Ph) = soft_dp(Cd, Cv, Ch, gamma)
    return Pv.sum() + Ph.sum()


def soft_divergence(x, y, gamma, kind="dtw", **params):
    """D(x, y) = S(x, y) - S(x, x)/2 - S(y, y)/2  (Blondel et al., 2021); D(x, x) = 0."""
    S = lambda a, b: soft_distance(a, b, gamma, kind, **params)
    return S(x, y) - 0.5 * S(x, x) - 0.5 * S(y, y)


def soft_divergence_value_and_grad(x, y, gamma, kind="dtw", **params):
    """Gradient wrt x; the distances are symmetric, so grad_x S(x, x) = 2 grad_1 S(x, x)."""
    vxy, gxy = soft_distance_value_and_grad(x, y, gamma, kind, **params)
    vxx, gxx = soft_distance_value_and_grad(x, x, gamma, kind, **params)
    vyy = soft_distance(y, y, gamma, kind, **params)
    return vxy - 0.5 * vxx - 0.5 * vyy, gxy - gxx
