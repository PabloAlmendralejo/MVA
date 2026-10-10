"""Barycenters for the soft distances of softdp.py: soft-DTW and soft-DTW_omega.

KINDS: 'dtw' (soft-DTW) and 'adtw' (soft-DTW with a warping penalty omega, i.e. a
soft version of amerced DTW).
"""

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from .softdp import soft_distance, soft_distance_value_and_grad, soft_dp_forward
from .barycenter import _weights
from .core import _as_2d


def barycenter_objective(z, Y, kind, gamma, omega=0.0, weights=None):
    """Barycenter objective sum_i S(z, y_i) / (N m_i) (Eq. 4) for S = soft-DTW or
    soft-DTW_omega, with weights w_i (uniform 1/N by default). Returns (value, gradient wrt z).
    """
    z = _as_2d(z)
    Y = [_as_2d(y) for y in Y]
    params = {"omega": omega} if kind == "adtw" else {}
    w = _weights(Y, weights)
    val, grad = 0.0, np.zeros_like(z)
    for wi, y in zip(w, Y):
        v, gr = soft_distance_value_and_grad(z, y, gamma, kind, **params)
        val += wi * v
        grad += wi * gr
    return val, grad


def soft_barycenter(Y, kind, gamma, init, omega=0.0, max_iter=100, weights=None):
    """L-BFGS barycenter for soft-DTW or soft-DTW_omega."""
    Y = [_as_2d(y) for y in Y]
    init = _as_2d(init)
    shape = init.shape

    def f(z):
        v, gr = barycenter_objective(z.reshape(shape), Y, kind, gamma, omega, weights)
        return v, gr.ravel()

    res = minimize(f, init.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x.reshape(shape)


@njit(parallel=True, cache=True)
def _pairwise_adtw(X, C, gamma, omega):
    A, n, _ = X.shape
    B, m, _ = C.shape
    out = np.empty((A, B))
    for a in prange(A):
        for b in range(B):
            Cd = np.full((n + 1, m + 1), np.inf)
            for i in range(n):
                for j in range(m):
                    Cd[i + 1, j + 1] = np.sum((X[a, i] - C[b, j]) ** 2)
            Cw = Cd + omega
            out[a, b] = soft_dp_forward(Cd, Cw, Cw, gamma)[n, m]
    return out


def pairwise_adtw(X, C, gamma, omega):
    """Matrix of soft-DTW_omega(X[a], C[b]) for equal-length series."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    if X.ndim == 2:
        X = X[..., None]
    if C.ndim == 2:
        C = C[..., None]
    return _pairwise_adtw(X, C, float(gamma), float(omega))
