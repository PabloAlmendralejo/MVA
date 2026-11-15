"""Shared Sinkhorn solver. Owner: person A.

This is the only piece of code the three parts share, so the *signature* is fixed
and the implementation is A's job. Convention:

    apply_K(v)[i] == sum_j K[i,j] * a[j] * v[j]

i.e. a function that applies the kernel to v, area weights included. `a` are the area
weights (cell size on a grid, lumped vertex areas on a mesh). Each kernel module only
has to provide its own apply_K and a.

Please don't change these signatures without telling the group.
"""

def sinkhorn(mu0, mu1, apply_K, a, gamma, n_iter=1000, tol=1e-6):
    v = np.ones(len(mu0))
    w = np.ones(len(mu1))

    for _ in range(n_iter):
        v = mu0 / apply_K(w)
        w = mu1 / apply_K(v)

        error = np.sum(a * np.abs(v * apply_K(w) - mu0))
        if error < tol:
            break

    p0, p1 = mu0 > 0, mu1 > 0
    dist2 = gamma * (np.sum(a[p0] * mu0[p0] * np.log(v[p0]))
                     + np.sum(a[p1] * mu1[p1] * np.log(w[p1])))
    return dist2, v, w


def barycenter(mus, alphas, apply_K, a, n_iter=500):
    """Wasserstein barycenter with weights `alphas` via iterated Bregman projections
    (Algorithm 2 of Solomon et al. 2015). Returns the barycentric density."""
    raise NotImplementedError
