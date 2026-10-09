"""Correctness checks for softdtw.py and torch_sdtw.py.  Run:  pytest tests.py -v"""

import itertools

import numpy as np
import pytest
import torch

from softdtw import (dtw, dtw_path, soft_dtw, soft_dtw_alignment,
                     soft_dtw_value_and_grad, sqeuclidean_cost)
from torch_sdtw import SoftDTWLoss, soft_dtw_torch_naive

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Ground truth by brute force: enumerate every alignment path
# ---------------------------------------------------------------------------

def all_paths(n, m):
    """Every monotone path (1,1) -> (n,m) with steps down, right, diagonal."""
    def rec(i, j):
        if (i, j) == (n - 1, m - 1):
            yield [(i, j)]
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            if i + di < n and j + dj < m:
                for tail in rec(i + di, j + dj):
                    yield [(i, j)] + tail
    return list(rec(0, 0))


def delannoy(a, b):
    return sum(2 ** k * _comb(a, k) * _comb(b, k) for k in range(min(a, b) + 1))


def _comb(n, k):
    from math import comb
    return comb(n, k)


def brute_force(D, gamma):
    """Exact soft-min over all paths and Gibbs-expected alignment (Eq. 1 and 3)."""
    n, m = D.shape
    costs, mats = [], []
    for path in all_paths(n, m):
        A = np.zeros((n, m))
        for i, j in path:
            A[i, j] = 1.0
        mats.append(A)
        costs.append((A * D).sum())
    costs = np.array(costs)
    if gamma == 0:
        return costs.min(), None
    z = -costs / gamma
    zmax = z.max()
    value = -gamma * (zmax + np.log(np.exp(z - zmax).sum()))
    w = np.exp(z - zmax)
    w /= w.sum()
    E = sum(wk * Ak for wk, Ak in zip(w, mats))
    return value, E


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_path_count_is_delannoy():
    for n, m in [(1, 1), (2, 3), (4, 4), (5, 3)]:
        assert len(all_paths(n, m)) == delannoy(n - 1, m - 1)


def test_worked_example_2x2():
    """Delta = [[1,3],[2,1]], gamma = 1: three paths with costs 2, 4, 5."""
    from softdtw import sdtw_backward, sdtw_forward
    D = np.array([[1.0, 3.0], [2.0, 1.0]])
    R = sdtw_forward(D, 1.0)
    expected = 1.0 - np.log(np.exp(-1) + np.exp(-4) + np.exp(-3))
    assert R[2, 2] == pytest.approx(expected, abs=1e-12)
    E = sdtw_backward(D, R, 1.0)
    p = np.exp(-np.array([2.0, 4.0, 5.0]))
    p /= p.sum()  # diagonal, via (2,1), via (1,2)
    assert np.allclose(E, [[1.0, p[2]], [p[1], 1.0]], atol=1e-12)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("n,m", [(1, 1), (1, 4), (3, 3), (4, 5), (5, 5)])
def test_matches_brute_force(n, m, gamma):
    x, y = rng.normal(size=(n, 2)), rng.normal(size=(m, 2))
    value, E = soft_dtw_alignment(x, y, gamma)
    bf_value, bf_E = brute_force(sqeuclidean_cost(x, y), gamma)
    assert value == pytest.approx(bf_value, rel=1e-10, abs=1e-10)
    assert np.allclose(E, bf_E, atol=1e-10)


@pytest.mark.parametrize("n,m", [(3, 3), (4, 5), (5, 2)])
def test_dtw_matches_brute_force(n, m):
    x, y = rng.normal(size=(n, 1)), rng.normal(size=(m, 1))
    assert dtw(x, y) == pytest.approx(brute_force(sqeuclidean_cost(x, y), 0)[0])


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_gradient_finite_differences(gamma):
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(11, 3))
    _, grad = soft_dtw_value_and_grad(x, y, gamma)
    eps = 1e-6
    fd = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd[idx] = (soft_dtw(xp, y, gamma) - soft_dtw(xm, y, gamma)) / (2 * eps)
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_gamma_to_zero_sandwich():
    """dtw - gamma log|A| <= dtw_gamma <= dtw, so dtw_gamma -> dtw as gamma -> 0."""
    x, y = rng.normal(size=(12, 1)), rng.normal(size=(15, 1))
    d0 = dtw(x, y)
    log_npaths = np.log(float(delannoy(11, 14)))
    for gamma in [1.0, 0.1, 0.01, 0.001]:
        s = soft_dtw(x, y, gamma)
        assert d0 - gamma * log_npaths - 1e-9 <= s <= d0 + 1e-9


def test_alignment_tends_to_dtw_path():
    x, y = rng.normal(size=(10, 1)), rng.normal(size=(13, 1))
    A, _ = dtw_path(x, y)
    _, E = soft_dtw_alignment(x, y, 1e-4)
    assert np.allclose(E, A, atol=1e-3)


def test_self_value_is_not_zero():
    """Known flaw (motivates the divergence): dtw_gamma(x, x) < 0 for gamma > 0."""
    x = rng.normal(size=(10, 1))
    assert soft_dtw(x, x, 1.0) < 0


def test_matches_tslearn():
    tsm = pytest.importorskip("tslearn.metrics")
    for gamma in [0.1, 1.0]:
        x, y = rng.normal(size=(20, 2)), rng.normal(size=(17, 2))
        assert soft_dtw(x, y, gamma) == pytest.approx(tsm.soft_dtw(x, y, gamma=gamma), rel=1e-8)
    # tslearn's dtw returns the square root of our (squared-cost) DTW
    x, y = rng.normal(size=(20, 1)), rng.normal(size=(17, 1))
    assert dtw(x, y) == pytest.approx(tsm.dtw(x, y) ** 2, rel=1e-8)


@pytest.mark.parametrize("gamma", [0.1, 1.0])
def test_torch_backward_matches_autograd(gamma):
    x = torch.tensor(rng.normal(size=(6, 2)), requires_grad=True)
    y = torch.tensor(rng.normal(size=(7, 2)), requires_grad=True)
    ref = soft_dtw_torch_naive(x, y, gamma)
    gx_ref, gy_ref = torch.autograd.grad(ref, (x, y))

    x2 = x.detach().clone().requires_grad_(True)
    y2 = y.detach().clone().requires_grad_(True)
    val = SoftDTWLoss(gamma, reduction="sum")(x2[None], y2[None])
    val.backward()
    assert val.item() == pytest.approx(ref.item(), rel=1e-10)
    assert torch.allclose(x2.grad, gx_ref, atol=1e-10)
    assert torch.allclose(y2.grad, gy_ref, atol=1e-10)


def test_torch_batch_and_chain_rule():
    """Batch of different pairs, mean reduction, and an outer factor (chain rule)."""
    B, gamma = 4, 0.5
    x = torch.tensor(rng.normal(size=(B, 9, 1)), requires_grad=True)
    y = torch.tensor(rng.normal(size=(B, 12, 1)))
    loss = 3.0 * SoftDTWLoss(gamma, reduction="mean")(x, y)
    loss.backward()
    for b in range(B):
        _, g = soft_dtw_value_and_grad(x[b].detach().numpy(), y[b].numpy(), gamma)
        assert np.allclose(x.grad[b].numpy(), 3.0 * g / B, atol=1e-10)


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("n,m", [(1, 1), (5, 9), (12, 7)])
def test_antidiagonal_torch_matches_numba(n, m, gamma):
    from torch_sdtw_diag import SoftDTWLossDiag
    B = 3
    x = torch.tensor(rng.normal(size=(B, n, 2)), requires_grad=True)
    y = torch.tensor(rng.normal(size=(B, m, 2)), requires_grad=True)
    x2, y2 = x.detach().clone().requires_grad_(True), y.detach().clone().requires_grad_(True)
    v1 = SoftDTWLoss(gamma, reduction="none")(x, y)
    v2 = SoftDTWLossDiag(gamma, reduction="none")(x2, y2)
    assert torch.allclose(v1, v2, rtol=1e-10, atol=1e-10)
    (v1 * torch.arange(1.0, B + 1)).sum().backward()
    (v2 * torch.arange(1.0, B + 1)).sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-9)
    assert torch.allclose(y.grad, y2.grad, atol=1e-9)


@pytest.mark.parametrize("omega", [0.0, 0.5])
@pytest.mark.parametrize("divergence", [False, True])
def test_torch_loss_omega_and_divergence_match_engine(omega, divergence):
    from softdp import (soft_distance, soft_distance_value_and_grad, soft_divergence,
                        soft_divergence_value_and_grad)
    kind, params = ("adtw", {"omega": omega}) if omega > 0 else ("dtw", {})
    B, gamma = 3, 0.3
    X, Y = rng.normal(size=(B, 7, 2)), rng.normal(size=(B, 9, 2))
    x = torch.tensor(X, requires_grad=True)
    v = SoftDTWLoss(gamma, reduction="none", omega=omega, divergence=divergence)(x, torch.tensor(Y))
    v.sum().backward()
    for b in range(B):
        f = soft_divergence_value_and_grad if divergence else soft_distance_value_and_grad
        ref_v, ref_g = f(X[b], Y[b], gamma, kind, **params)
        assert v[b].item() == pytest.approx(ref_v, rel=1e-10, abs=1e-10)
        assert np.allclose(x.grad[b].numpy(), ref_g, atol=1e-9)
