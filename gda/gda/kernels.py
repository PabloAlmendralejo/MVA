"""Heat-kernel approximations of exp(-d(x,y)^2 / gamma).

Each function returns `(apply_K, a)` following the convention in sinkhorn.py:

    apply_K(v)[i] == sum_j K[i,j] * a[j] * v[j]

Owners: the grid kernel is B's, the mesh kernel is C's.
"""


def gaussian_grid_kernel(shape, gamma, h=1.0):
    """Regular grid: Gaussian convolution, i.e. the exact heat kernel at time gamma/2.
    `shape` is the grid shape, `h` the grid spacing. Owner: B."""
    raise NotImplementedError


def heat_kernel(L, a, gamma, substeps=1):
    solve = spla.factorized(sp.csc_matrix(sp.diags(a) + gamma / (4 * substeps) * L))

    def apply_K(v):
        for _ in range(substeps):
            v = solve(a * v)
        return v
    return apply_K
