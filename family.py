"""Fast pairwise evaluation and barycenters for the soft-distance family of softdp.py.

KINDS:  'dtw' (soft-DTW), 'adtw' (soft amerced DTW, warping penalty omega),
        'wdtw' (soft weighted DTW, logistic weight with steepness g),
        'erp'  (soft edit distance with real penalty, gap value 0),
plus two order-free / warping-free controls:
        'euclidean' (equal lengths) and 'wasserstein' (1D OT between the value
        distributions: sorts each series, so it ignores time completely).
Each soft distance can be used raw or debiased (divergence=True).
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from softdp import _softmin3, soft_distance_value_and_grad, soft_divergence_value_and_grad
from softdtw import _as_2d

KIND_ID = {"dtw": 0, "adtw": 1, "wdtw": 2, "erp": 3}


@njit(cache=True)
def _pair_value(x, y, gamma, kind, omega, g):
    """soft_dp_forward specialised per kind, without building cost arrays (gap = 0)."""
    n, m = x.shape[0], y.shape[0]
    half = max(n, m) / 2.0
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    if kind == 3:  # ERP borders: matching against gaps
        for i in range(1, n + 1):
            R[i, 0] = R[i - 1, 0] + (x[i - 1] ** 2).sum()
        for j in range(1, m + 1):
            R[0, j] = R[0, j - 1] + (y[j - 1] ** 2).sum()
    for i in range(1, n + 1):
        gx = (x[i - 1] ** 2).sum()
        for j in range(1, m + 1):
            d = ((x[i - 1] - y[j - 1]) ** 2).sum()
            if kind == 0:
                cd = cv = ch = d
            elif kind == 1:
                cd, cv, ch = d, d + omega, d + omega
            elif kind == 2:
                w = 1.0 / (1.0 + np.exp(-g * (abs(i - j) - half)))
                cd = cv = ch = w * d
            else:
                cd, cv, ch = d, gx, (y[j - 1] ** 2).sum()
            R[i, j] = _softmin3(R[i - 1, j - 1] + cd, R[i - 1, j] + cv, R[i, j - 1] + ch, gamma)
    return R[n, m]


@njit(parallel=True, cache=True)
def _pairwise(A, B, gamma, kind, omega, g):
    out = np.empty((A.shape[0], B.shape[0]))
    for a in prange(A.shape[0]):
        for b in range(B.shape[0]):
            out[a, b] = _pair_value(A[a], B[b], gamma, kind, omega, g)
    return out


@njit(parallel=True, cache=True)
def _selfvals(A, gamma, kind, omega, g):
    out = np.empty(A.shape[0])
    for a in prange(A.shape[0]):
        out[a] = _pair_value(A[a], A[a], gamma, kind, omega, g)
    return out


def _3d(A):
    A = np.ascontiguousarray(A, dtype=np.float64)
    return A[..., None] if A.ndim == 2 else A


def pairwise(A, B, kind, gamma=1.0, omega=0.0, g=0.05, divergence=False):
    """Matrix of distances between series A[a] and B[b]."""
    A, B = _3d(A), _3d(B)
    if kind == "euclidean":
        return ((A[:, None] - B[None]) ** 2).sum((2, 3))
    if kind == "wasserstein":  # 1D OT between value distributions (order ignored)
        sa, sb = np.sort(A[..., 0], 1), np.sort(B[..., 0], 1)
        return ((sa[:, None] - sb[None]) ** 2).sum(2)
    k = KIND_ID[kind]
    D = _pairwise(A, B, float(gamma), k, float(omega), float(g))
    if divergence:
        D = D - 0.5 * _selfvals(A, gamma, k, omega, g)[:, None] - 0.5 * _selfvals(B, gamma, k, omega, g)[None, :]
    return D


def family_barycenter(Y, kind, gamma, init, omega=0.0, g=0.05, divergence=False, max_iter=100):
    """L-BFGS barycenter for any member of the family (raw or debiased)."""
    Y = [_as_2d(y) for y in Y]
    init = _as_2d(init)
    shape, N = init.shape, len(Y)
    params = {"adtw": {"omega": omega}, "wdtw": {"g": g}}.get(kind, {})
    vg = soft_divergence_value_and_grad if divergence else soft_distance_value_and_grad

    def f(z):
        z = z.reshape(shape)
        val, grad = 0.0, np.zeros(shape)
        for y in Y:
            v, gr = vg(z, y, gamma, kind, **params)
            val += v / (N * len(y))
            grad += gr / (N * len(y))
        return val, grad.ravel()

    res = minimize(f, init.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x.reshape(shape)
