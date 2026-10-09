"""Checks for the general soft DP engine (softdp.py).  Run:  pytest tests_softdp.py -v"""

import numpy as np
import pytest

from softdp import (edge_costs, expected_warps, soft_distance,
                    soft_distance_value_and_grad, soft_divergence,
                    soft_divergence_value_and_grad, soft_dp, soft_dp_forward)
from softdtw import soft_dtw, soft_dtw_alignment

rng = np.random.default_rng(1)
KINDS = [("dtw", {}), ("adtw", {"omega": 0.7}), ("wdtw", {"g": 0.3}), ("erp", {"gap": 0.2})]


def all_grid_paths(n, m):
    """Every path (0,0) -> (n,m) on the node grid, as a list of (edge_type, i, j)."""
    def rec(i, j):
        if (i, j) == (n, m):
            yield []
            return
        for t, di, dj in (("d", 1, 1), ("v", 1, 0), ("h", 0, 1)):
            if i + di <= n and j + dj <= m:
                for tail in rec(i + di, j + dj):
                    yield [(t, i + di, j + dj)] + tail
    return list(rec(0, 0))


def brute_force(Cd, Cv, Ch, gamma):
    n, m = Cd.shape[0] - 1, Cd.shape[1] - 1
    C = {"d": Cd, "v": Cv, "h": Ch}
    costs, uses = [], []
    for path in all_grid_paths(n, m):
        costs.append(sum(C[t][i, j] for t, i, j in path))
        U = {t: np.zeros_like(Cd) for t in C}
        for t, i, j in path:
            U[t][i, j] += 1
        uses.append(U)
    costs = np.array(costs)
    ok = np.isfinite(costs)
    z = -costs[ok] / gamma
    zmax = z.max()
    w = np.exp(z - zmax)
    value = -gamma * (zmax + np.log(w.sum()))
    w /= w.sum()
    used = [u for u, k in zip(uses, ok) if k]
    P = tuple(sum(wk * u[t] for wk, u in zip(w, used)) for t in "dvh")
    return value, P


@pytest.mark.parametrize("kind,params", KINDS)
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("n,m", [(1, 1), (2, 4), (4, 3)])
def test_engine_matches_brute_force(kind, params, gamma, n, m):
    x, y = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    Cd, Cv, Ch = edge_costs(x, y, kind, **params)
    value, P = soft_dp(Cd, Cv, Ch, gamma)
    bf_value, bf_P = brute_force(Cd, Cv, Ch, gamma)
    assert value == pytest.approx(bf_value, rel=1e-10, abs=1e-10)
    for a, b in zip(P, bf_P):
        assert np.allclose(a, b, atol=1e-10)


def test_dtw_case_is_soft_dtw():
    x, y = rng.normal(size=(9, 2)), rng.normal(size=(12, 2))
    Cd, Cv, Ch = edge_costs(x, y, "dtw")
    value, (Pd, Pv, Ph) = soft_dp(Cd, Cv, Ch, 0.5)
    ref_value, E = soft_dtw_alignment(x, y, 0.5)
    assert value == pytest.approx(ref_value, rel=1e-12)
    assert np.allclose((Pd + Pv + Ph)[1:, 1:], E, atol=1e-12)
    assert soft_distance(x, y, 0.0, "dtw") == pytest.approx(soft_dtw(x, y, 0.0))


def test_adtw_limits():
    """omega = 0 gives soft-DTW; omega -> inf forbids warping (equal lengths: Euclidean)."""
    x, y = rng.normal(size=(10, 1)), rng.normal(size=(10, 1))
    assert soft_distance(x, y, 0.3, "adtw", omega=0.0) == pytest.approx(soft_dtw(x, y, 0.3))
    eucl = ((x - y) ** 2).sum()
    assert soft_distance(x, y, 0.3, "adtw", omega=1e6) == pytest.approx(eucl, rel=1e-9)


@pytest.mark.parametrize("kind,params", KINDS)
@pytest.mark.parametrize("divergence", [False, True])
def test_gradient_finite_differences(kind, params, divergence):
    x, y = rng.normal(size=(7, 2)), rng.normal(size=(9, 2))
    gamma = 0.3
    if divergence:
        f = lambda z: soft_divergence(z, y, gamma, kind, **params)
        _, grad = soft_divergence_value_and_grad(x, y, gamma, kind, **params)
    else:
        f = lambda z: soft_distance(z, y, gamma, kind, **params)
        _, grad = soft_distance_value_and_grad(x, y, gamma, kind, **params)
    eps, fd = 1e-6, np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd[idx] = (f(xp) - f(xm)) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_omega_derivative_is_expected_warps():
    x, y = rng.normal(size=(8, 1)), rng.normal(size=(11, 1))
    gamma, omega, eps = 0.5, 0.4, 1e-6
    fd = (soft_distance(x, y, gamma, "adtw", omega=omega + eps)
          - soft_distance(x, y, gamma, "adtw", omega=omega - eps)) / (2 * eps)
    assert expected_warps(x, y, gamma, omega) == pytest.approx(fd, rel=1e-6)


@pytest.mark.parametrize("kind,params", KINDS)
def test_divergence_is_zero_on_identical_series(kind, params):
    x = rng.normal(size=(10, 1))
    assert soft_distance(x, x, 1.0, kind, **params) != pytest.approx(0.0)
    assert soft_divergence(x, x, 1.0, kind, **params) == pytest.approx(0.0, abs=1e-12)


def test_hard_forward_equals_min_over_paths():
    x, y = rng.normal(size=(4, 1)), rng.normal(size=(3, 1))
    for kind, params in KINDS:
        Cd, Cv, Ch = edge_costs(x, y, kind, **params)
        hard = soft_dp_forward(Cd, Cv, Ch, 0.0)[-1, -1]
        C = {"d": Cd, "v": Cv, "h": Ch}
        best = min(sum(C[t][i, j] for t, i, j in p) for p in all_grid_paths(4, 3))
        assert hard == pytest.approx(best)


def test_fast_pairwise_matches_engine():
    from family import pairwise
    from softdp import soft_divergence
    A, B = rng.normal(size=(3, 8, 1)), rng.normal(size=(4, 10, 1))
    for kind, params in [("dtw", {}), ("adtw", {"omega": 0.7}), ("wdtw", {"g": 0.3}), ("erp", {})]:
        for div in (False, True):
            M = pairwise(A, B, kind, gamma=0.4, divergence=div, **params)
            f = soft_divergence if div else soft_distance
            ref = np.array([[f(a, b, 0.4, kind, **params) for b in B] for a in A])
            assert np.allclose(M, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind,params", KINDS)
@pytest.mark.parametrize("divergence", [False, True])
def test_barycenter_objective_gradient(kind, params, divergence):
    from family import family_objective
    Y = [rng.normal(size=(6, 1)), rng.normal(size=(8, 1)), rng.normal(size=(7, 1))]
    z = rng.normal(size=(5, 1))
    kw = dict(omega=params.get("omega", 0.0), g=params.get("g", 0.05), divergence=divergence)
    _, grad = family_objective(z, Y, kind, 0.3, **kw)
    eps, fd = 1e-6, np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        fd[idx] = (family_objective(zp, Y, kind, 0.3, **kw)[0] - family_objective(zm, Y, kind, 0.3, **kw)[0]) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-7)


def test_family_barycenter_dtw_matches_softdtw_barycenter():
    from barycenter import softdtw_barycenter
    from family import family_barycenter
    Y = [rng.normal(size=(12, 1)) for _ in range(4)]
    init = rng.normal(size=(12, 1))
    a = family_barycenter(Y, "dtw", 0.1, init)
    b = softdtw_barycenter(Y, 0.1, init)
    assert np.allclose(a, b, atol=1e-6)
