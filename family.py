"""Barycenters for the soft distances of softdp.py.

KINDS: 'dtw' (soft-DTW) and 'adtw' (soft-DTW with a warping penalty omega, i.e. a
soft version of amerced DTW). Each can be used raw or debiased (divergence=True).
"""

import numpy as np
from scipy.optimize import minimize

from softdp import soft_distance_value_and_grad
from softdtw import _as_2d


def family_objective(z, Y, kind, gamma, omega=0.0, divergence=False):
    """Barycenter objective sum_i S(z, y_i) / (N m_i) (Eq. 4) for S = soft-DTW or
    soft-DTW_omega, or its debiased version with D(z, y) = S(z, y) - S(z, z)/2 - S(y, y)/2.

    The S(y, y) terms do not depend on z and are dropped (same minimiser).
    Returns (value, gradient wrt z).
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
    if divergence:
        # grad_z S(z, z) = 2 grad_1 S(z, z) because the distances are symmetric
        vzz, gzz = soft_distance_value_and_grad(z, z, gamma, kind, **params)
        val -= 0.5 * w.sum() * vzz
        grad -= w.sum() * gzz
    return val, grad


def family_barycenter(Y, kind, gamma, init, omega=0.0, divergence=False, max_iter=100):
    """L-BFGS barycenter for soft-DTW or soft-DTW_omega (raw or debiased)."""
    Y = [_as_2d(y) for y in Y]
    init = _as_2d(init)
    shape = init.shape

    def f(z):
        v, gr = family_objective(z.reshape(shape), Y, kind, gamma, omega, divergence)
        return v, gr.ravel()

    res = minimize(f, init.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x.reshape(shape)
