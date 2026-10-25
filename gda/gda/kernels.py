"""Heat-kernel approximations of exp(-d(x,y)^2 / gamma).

Each function returns `(apply_K, a)` following the convention in sinkhorn.py:

    apply_K(v)[i] == sum_j K[i,j] * a[j] * v[j]

Owners: the grid kernel is B's, the mesh kernel is C's.
"""


def gaussian_grid_kernel(shape, gamma, h=1.0):
    """Regular grid: Gaussian convolution, i.e. the exact heat kernel at time gamma/2.
    `shape` is the grid shape, `h` the grid spacing. Owner: B."""
    raise NotImplementedError


def implicit_heat_kernel(L, areas, gamma, substeps=1):
    """Mesh / graph: one implicit Euler step of the heat equation,
    (D_a + t L) u = D_a v with t = gamma / (2 * substeps), prefactored once.
    `L` is a positive semi-definite (cotangent) Laplacian. Owner: C."""
    raise NotImplementedError
