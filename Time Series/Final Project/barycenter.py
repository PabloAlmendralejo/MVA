"""Barycenters of time series under (soft-)DTW (Section 3.1 of the paper).

Objective of Eq. (4):  F_gamma(x) = sum_i  w_i / m_i * dtw_gamma(x, y_i),
with uniform weights w_i = 1/N by default. All methods are evaluated with the
*hard* DTW loss F_0, as in Table 1 of the paper.

Methods
-------
softdtw_barycenter   : L-BFGS on F_gamma (the paper's proposal)
subgradient_barycenter: batch subgradient descent on F_0 (paper's baseline);
                        step chosen by Armijo backtracking, best iterate kept
dba_barycenter       : DBA (Petitjean et al., 2011), tslearn implementation
"""

import numpy as np
from scipy.optimize import minimize

from softdtw import (_as_2d, dtw, dtw_value_and_subgrad, soft_dtw_value_and_grad)


def _weights(Y, weights):
    N = len(Y)
    w = np.full(N, 1.0 / N) if weights is None else np.asarray(weights, float)
    return w / np.array([len(y) for y in Y])  # w_i / m_i


def dtw_barycenter_loss(x, Y, weights=None):
    """F_0(x): the evaluation metric of Table 1 (hard DTW, Eq. 4 with gamma = 0)."""
    c = _weights(Y, weights)
    return float(sum(ci * dtw(x, y) for ci, y in zip(c, Y)))


def softdtw_objective(x, Y, gamma, weights=None):
    """F_gamma(x) and its gradient."""
    x = _as_2d(x)
    c = _weights(Y, weights)
    value, grad = 0.0, np.zeros_like(x)
    for ci, y in zip(c, Y):
        v, g = soft_dtw_value_and_grad(x, y, gamma)
        value += ci * v
        grad += ci * g
    return value, grad


def softdtw_barycenter(Y, gamma, init, weights=None, max_iter=100):
    """Minimise F_gamma with L-BFGS, starting from `init` (shape (n, p))."""
    Y = [_as_2d(y) for y in Y]
    init = _as_2d(init)
    shape = init.shape

    def f(z):
        v, g = softdtw_objective(z.reshape(shape), Y, gamma, weights)
        return v, g.ravel()

    res = minimize(f, init.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    return res.x.reshape(shape)


def subgradient_barycenter(Y, init, weights=None, max_iter=100, step0=1.0):
    """Batch subgradient descent on the hard DTW objective F_0.

    Each iteration uses the subgradient of Eq. (2) and a backtracking step
    (halve until the objective decreases, at most 20 halvings). The best
    iterate is returned, as is standard for subgradient methods.
    """
    Y = [_as_2d(y) for y in Y]
    x = _as_2d(init).copy()
    c = _weights(Y, weights)

    def obj_and_subgrad(z):
        v, g = 0.0, np.zeros_like(z)
        for ci, y in zip(c, Y):
            vi, gi = dtw_value_and_subgrad(z, y)
            v += ci * vi
            g += ci * gi
        return v, g

    value, g = obj_and_subgrad(x)
    best_x, best_v = x.copy(), value
    step = step0
    for _ in range(max_iter):
        gnorm2 = (g ** 2).sum()
        if gnorm2 == 0:
            break
        t = step
        for _ in range(20):
            x_new = x - t * g
            v_new = dtw_barycenter_loss(x_new, Y, weights)
            if v_new <= value - 1e-4 * t * gnorm2:
                break
            t *= 0.5
        else:
            break  # no decrease found: stop
        x = x_new
        value, g = obj_and_subgrad(x)
        step = min(2 * t, step0)  # allow the step to grow back
        if value < best_v:
            best_x, best_v = x.copy(), value
    return best_x


def dba_barycenter(Y, init, weights=None, max_iter=100):
    """DBA (Petitjean et al. 2011) from tslearn, same init and iteration budget."""
    from tslearn.barycenters import dtw_barycenter_averaging_petitjean
    Y = [_as_2d(y) for y in Y]
    return dtw_barycenter_averaging_petitjean(
        Y, init_barycenter=_as_2d(init), max_iter=max_iter, tol=1e-12, weights=weights)


def euclidean_mean(Y):
    """Arithmetic mean (only for equal lengths): the 'Euclidean mean' init."""
    return np.mean([_as_2d(y) for y in Y], axis=0)
