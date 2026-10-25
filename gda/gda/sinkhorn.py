"""Shared Sinkhorn solver. Owner: person A.

This is the only piece of code the three parts share, so the *signature* is fixed
and the implementation is A's job. Convention:

    apply_K(v)[i] == sum_j K[i,j] * a[j] * v[j]

i.e. a function that applies the kernel to v, area weights included. `a` are the area
weights (cell size on a grid, lumped vertex areas on a mesh). Each kernel module only
has to provide its own apply_K and a.

Please don't change these signatures without telling the group.
"""


def sinkhorn(mu0, mu1, apply_K, a, gamma, n_iter=500, tol=1e-9):
    """Entropic OT between mu0 and mu1 (Algorithm 1 of Solomon et al. 2015).

    Returns (dist2, v, w), with the transport plan pi = diag(v) K diag(w) and
    dist2 = gamma * sum_i a_i (mu0_i log v_i + mu1_i log w_i).
    """
    raise NotImplementedError


def barycenter(mus, alphas, apply_K, a, n_iter=500):
    """Wasserstein barycenter with weights `alphas` via iterated Bregman projections
    (Algorithm 2 of Solomon et al. 2015). Returns the barycentric density."""
    raise NotImplementedError
