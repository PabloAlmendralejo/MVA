"""Soft-DTW (Cuturi & Blondel, ICML 2017) -- reference implementation.

Conventions
-----------
A time series is an array of shape (n, p): n time steps, p features
(the paper uses p x n; we transpose to match numpy / tslearn).
Univariate series of shape (n,) are accepted and treated as (n, 1).

Main entry points
-----------------
sqeuclidean_cost(x, y)          -> Delta, the n x m cost matrix
sdtw_forward(D, gamma)          -> R, accumulated soft costs (Algorithm 1)
sdtw_backward(D, R, gamma)      -> E, expected alignment matrix (Algorithm 2)
soft_dtw(x, y, gamma)           -> value of dtw_gamma(x, y)
soft_dtw_value_and_grad(x, y, gamma) -> (value, grad wrt x)
soft_dtw_alignment(x, y, gamma) -> (value, E)
dtw(x, y), dtw_path(x, y)       -> classic DTW (gamma = 0) and its optimal path
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_2d(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("time series must have shape (n,) or (n, p)")
    return x


def sqeuclidean_cost(x, y):
    """Delta[i, j] = ||x_i - y_j||^2, shape (n, m)."""
    x, y = _as_2d(x), _as_2d(y)
    sq = (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2.0 * x @ y.T
    return np.maximum(sq, 0.0)  # clip tiny negative round-off


@njit(cache=True)
def _softmin3(a, b, c, gamma):
    """-gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma)), stable."""
    m = min(a, b, c)
    if m == np.inf:
        return np.inf
    s = np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    return m - gamma * np.log(s)


# ---------------------------------------------------------------------------
# Algorithm 1: forward recursion
# ---------------------------------------------------------------------------

@njit(cache=True)
def sdtw_forward(D, gamma):
    """Bellman recursion of soft-DTW.

    Returns R of shape (n + 2, m + 2) with 1-based interior R[1:n+1, 1:m+1];
    R[n, m] is dtw_gamma. Row/column 0 hold the +inf borders of Algorithm 1;
    row/column n+1, m+1 are left for the backward pass. gamma = 0 gives DTW.
    """
    n, m = D.shape
    R = np.full((n + 2, m + 2), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gamma == 0.0:
                prev = min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
            else:
                prev = _softmin3(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], gamma)
            R[i, j] = D[i - 1, j - 1] + prev
    return R


# ---------------------------------------------------------------------------
# Algorithm 2: backward recursion
# ---------------------------------------------------------------------------

@njit(cache=True)
def sdtw_backward(D, R, gamma):
    """Backward pass: E[i, j] = d r_{n,m} / d r_{i,j} = E_gamma[A]_{ij}.

    D is the (n, m) cost matrix, R the output of sdtw_forward (not modified).
    Requires gamma > 0.
    """
    n, m = D.shape
    # padded cost: D_[i, j] = delta_{i,j} (1-based), zeros on the extra border
    D_ = np.zeros((n + 2, m + 2))
    D_[1:n + 1, 1:m + 1] = D
    # extended R: -inf on row n+1 / column m+1 so exp(.) = 0 there,
    # and r_{n+1,m+1} = r_{n,m} so that the last cell gets weight exactly 1
    R_ = R.copy()
    R_[:, m + 1] = -np.inf
    R_[n + 1, :] = -np.inf
    R_[n + 1, m + 1] = R[n, m]
    E = np.zeros((n + 2, m + 2))
    E[n + 1, m + 1] = 1.0
    for i in range(n, 0, -1):  # row by row: follows the memory layout
        for j in range(m, 0, -1):
            a = np.exp((R_[i + 1, j] - R_[i, j] - D_[i + 1, j]) / gamma)
            b = np.exp((R_[i, j + 1] - R_[i, j] - D_[i, j + 1]) / gamma)
            c = np.exp((R_[i + 1, j + 1] - R_[i, j] - D_[i + 1, j + 1]) / gamma)
            E[i, j] = a * E[i + 1, j] + b * E[i, j + 1] + c * E[i + 1, j + 1]
    return E[1:n + 1, 1:m + 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def soft_dtw(x, y, gamma=1.0):
    """dtw_gamma(x, y). gamma = 0 returns classic DTW (sum of squared costs)."""
    D = sqeuclidean_cost(x, y)
    R = sdtw_forward(D, float(gamma))
    return R[D.shape[0], D.shape[1]]


def soft_dtw_alignment(x, y, gamma=1.0):
    """Returns (dtw_gamma(x, y), E) where E is the expected alignment matrix."""
    if gamma <= 0:
        raise ValueError("soft alignment requires gamma > 0; use dtw_path for gamma = 0")
    D = sqeuclidean_cost(x, y)
    R = sdtw_forward(D, float(gamma))
    E = sdtw_backward(D, R, float(gamma))
    return R[D.shape[0], D.shape[1]], E


def jacobian_product_sqeuclidean(x, y, E):
    """(d Delta / d x)^T E for Delta = squared Euclidean.

    Row i: sum_j E[i, j] * 2 (x_i - y_j)  =  2 (E.sum(1)[:, None] * x - E @ y).
    """
    x, y = _as_2d(x), _as_2d(y)
    return 2.0 * (E.sum(1)[:, None] * x - E @ y)


_workspace = {}


def _buffers(n, m):
    """Padded cost, R, E and a matmul buffer, reused across calls of the same shape
    (one shape at a time: series of a dataset share their length)."""
    if (n, m) not in _workspace:
        _workspace.clear()
        _workspace[(n, m)] = (np.zeros((n + 2, m + 2)), np.empty((n + 2, m + 2)),
                              np.empty((n + 2, m + 2)), np.empty((n, m)))
    return _workspace[(n, m)]


@njit(cache=True)
def _value_and_alignment(D_, R, E, n, m, gamma):
    """sdtw_forward + sdtw_backward in place (same arithmetic); D_ is the zero-padded cost."""
    R[0, :] = np.inf
    R[:, 0] = np.inf
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            R[i, j] = D_[i, j] + _softmin3(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], gamma)
    value = R[n, m]
    R[:, m + 1] = -np.inf
    R[n + 1, :] = -np.inf
    R[n + 1, m + 1] = value
    E[n + 1, :] = 0.0
    E[:, m + 1] = 0.0
    E[n + 1, m + 1] = 1.0
    for i in range(n, 0, -1):
        for j in range(m, 0, -1):
            a = np.exp((R[i + 1, j] - R[i, j] - D_[i + 1, j]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D_[i, j + 1]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D_[i + 1, j + 1]) / gamma)
            E[i, j] = a * E[i + 1, j] + b * E[i, j + 1] + c * E[i + 1, j + 1]
    return value


def soft_dtw_value_and_grad(x, y, gamma=1.0):
    """Returns (dtw_gamma(x, y), gradient wrt x of shape (n, p)).

    Bit-identical to soft_dtw_alignment + jacobian_product_sqeuclidean, without
    allocating the n x m matrices at every call (this is the inner loop of the barycenters).
    """
    if gamma <= 0:
        raise ValueError("soft alignment requires gamma > 0; use dtw_path for gamma = 0")
    x, y = _as_2d(x), _as_2d(y)
    n, m = x.shape[0], y.shape[0]
    D_, R, E, M = _buffers(n, m)
    D = D_[1:n + 1, 1:m + 1]  # sqeuclidean_cost, written in place
    np.add((x ** 2).sum(1)[:, None], (y ** 2).sum(1)[None, :], out=D)
    np.matmul(2.0 * x, y.T, out=M)
    np.subtract(D, M, out=D)
    np.maximum(D, 0.0, out=D)
    value = _value_and_alignment(D_, R, E, n, m, float(gamma))
    return value, jacobian_product_sqeuclidean(x, y, E[1:n + 1, 1:m + 1])


def dtw(x, y):
    """Classic DTW with squared Euclidean cost (no square root)."""
    return soft_dtw(x, y, gamma=0.0)


def dtw_path(x, y):
    """Optimal alignment of classic DTW as a binary (n, m) matrix A*, plus its cost."""
    D = sqeuclidean_cost(x, y)
    R = sdtw_forward(D, 0.0)
    n, m = D.shape
    A = np.zeros((n, m))
    i, j = n, m
    while True:
        A[i - 1, j - 1] = 1.0
        if i == 1 and j == 1:
            break
        candidates = ((R[i - 1, j - 1], i - 1, j - 1),
                      (R[i - 1, j], i - 1, j),
                      (R[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda t: t[0])
    return A, R[n, m]


def dtw_value_and_subgrad(x, y):
    """DTW value and the (sub)gradient wrt x of Eq. (2): (d Delta / d x)^T A*."""
    x, y = _as_2d(x), _as_2d(y)
    A, value = dtw_path(x, y)
    return value, jacobian_product_sqeuclidean(x, y, A)
