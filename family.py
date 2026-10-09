"""Barycenters for the soft distances of softdp.py.

KINDS: 'dtw' (soft-DTW) and 'adtw' (soft-DTW with a warping penalty omega, i.e. a
soft version of amerced DTW).
"""

import numpy as np
from scipy.optimize import minimize

from softdp import soft_distance_value_and_grad
from softdtw import _as_2d


def family_objective(z, Y, kind, gamma, omega=0.0):
    """Barycenter objective sum_i S(z, y_i) / (N m_i) (Eq. 4) for S = soft-DTW or
    soft-DTW_omega. Returns (value, gradient wrt z).
    """
    z = _as_2d(z)
    Y = [_as_2d(y) for y in Y]
    params = {"omega": omega} if kind == "adtw" else {}
    w = np.array([1.0 / (len(Y) * len(y)) for y in Y])
    val, grad = 0.0, np.zeros_like(z)
    for wi, y in zip(w, Y):
        v, gr = soft_distance_value_and_grad(z, y, gamma, kind, **params)
        val += wi * v
        grad += wi * gr
    return val, grad


def family_barycenter(Y, kind, gamma, init, omega=0.0, max_iter=100):
    """L-BFGS barycenter for soft-DTW or soft-DTW_omega."""
    Y = [_as_2d(y) for y in Y]
    init = _as_2d(init)
    shape = init.shape

    def f(z):
        v, gr = family_objective(z.reshape(shape), Y, kind, gamma, omega)
        return v, gr.ravel()

    res = minimize(f, init.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x.reshape(shape)
