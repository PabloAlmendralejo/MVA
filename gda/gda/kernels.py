def gaussian_grid_kernel(shape, gamma, h=1.0):
    n1, n2 = shape

    def gauss_1d(m):
        x = h * np.arange(m)
        return np.exp(-(x[:, None] - x[None, :]) ** 2 / gamma) / np.sqrt(np.pi * gamma)

    G1, G2 = gauss_1d(n1), gauss_1d(n2)
    a = h * h

    def apply_K(v):
        V = (a * v).reshape(n1, n2)
        return (G1 @ V @ G2.T).ravel()

    return apply_K


def heat_kernel(L, a, gamma, substeps=1):
    solve = spla.factorized(sp.csc_matrix(sp.diags(a) + gamma / (4 * substeps) * L))

    def apply_K(v):
        for _ in range(substeps):
            v = solve(a * v)
        return v
    return apply_K
