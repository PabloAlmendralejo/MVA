"""k-means (Section 3.2) and nearest-centroid classification (Section 3.3) under (soft-)DTW.

k-means energy of Eq. (5):  sum_i  1/m_i * min_j dtw_gamma(c_j, y_i);
methods are compared with the hard energy (gamma = 0), as in Table 2.
"""

import numpy as np
from numba import njit, prange

from barycenter import dba_barycenter, softdtw_barycenter, subgradient_barycenter
from omega import soft_barycenter, pairwise_adtw
from softdtw import sdtw_forward, sqeuclidean_cost


@njit(parallel=True, cache=True)
def _pairwise(X, C, gamma):
    """dtw_gamma between every series X[a] (N, n, p) and centroid C[b] (k, n', p)."""
    N, k = X.shape[0], C.shape[0]
    out = np.empty((N, k))
    for a in prange(N):
        for b in range(k):
            x, c = X[a], C[b]
            D = (x ** 2).sum(1).reshape(-1, 1) + (c ** 2).sum(1).reshape(1, -1) - 2.0 * x @ c.T
            D = np.maximum(D, 0.0)
            R = sdtw_forward(D, gamma)
            out[a, b] = R[x.shape[0], c.shape[0]]
    return out


def pairwise_sdtw(X, C, gamma):
    """Matrix of dtw_gamma(X[a], C[b]); gamma = 0 gives hard DTW."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    if X.ndim == 2:
        X = X[..., None]
    if C.ndim == 2:
        C = C[..., None]
    return _pairwise(X, C, float(gamma))


def kmeans_energy(X, C):
    """Hard-DTW k-means energy (Eq. 5 with gamma = 0), normalised by series length."""
    return float(pairwise_sdtw(X, C, 0.0).min(1).sum() / X.shape[1])


def dtw_kmeans(X, k, method, init_centroids, gamma=None, max_outer=30, max_inner=100, omega=0.0):
    """Lloyd's algorithm where assignment and centring both use the method's geometry.

    method: 'soft-DTW' (assign with dtw_gamma, centre with soft-DTW barycenter),
            'soft-DTW_omega' (assign and centre with soft-DTW_omega, penalty omega),
            'DBA' or 'subgradient' (assign with hard DTW, centre with that barycenter).
    Returns (centroids, labels).
    """
    C = np.array(init_centroids, dtype=np.float64)
    assign_gamma = gamma if method == "soft-DTW" else 0.0
    labels = None
    for _ in range(max_outer):
        if method == "soft-DTW_omega":
            new_labels = pairwise_adtw(X, C, gamma, omega).argmin(1)
        else:
            new_labels = pairwise_sdtw(X, C, assign_gamma).argmin(1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = [X[i] for i in np.where(labels == j)[0]]
            if not members:  # empty cluster: keep previous centroid
                continue
            if method == "soft-DTW":
                C[j] = softdtw_barycenter(members, gamma, C[j], max_iter=max_inner)
            elif method == "soft-DTW_omega":
                C[j] = soft_barycenter(members, "adtw", gamma, C[j], omega=omega,
                                         max_iter=max_inner).reshape(C[j].shape)
            elif method == "DBA":
                C[j] = dba_barycenter(members, C[j], max_iter=max_inner)
            else:
                C[j] = subgradient_barycenter(members, C[j], max_iter=max_inner)
    return C, labels


def nearest_centroid_fit(X, y, method, gamma=None, max_iter=100):
    """One barycenter per class (init: Euclidean class mean)."""
    classes = np.unique(y)
    C = []
    for c in classes:
        members = list(X[y == c])
        init = np.mean(members, axis=0)
        if method == "soft-DTW":
            C.append(softdtw_barycenter(members, gamma, init, max_iter=max_iter))
        elif method == "DBA":
            C.append(dba_barycenter(members, init, max_iter=max_iter))
        else:
            C.append(init)  # Euclidean mean
    return classes, np.array(C)


def nearest_centroid_predict(X, classes, C, gamma=0.0):
    """Assign each series to the class whose centroid is closest in dtw_gamma."""
    return classes[pairwise_sdtw(X, C, gamma).argmin(1)]
