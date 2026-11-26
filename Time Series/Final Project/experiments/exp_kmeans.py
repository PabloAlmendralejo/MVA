"""Table 2 of the paper: k-means under soft-DTW vs DBA vs subgradient.

k = number of classes; at most 30 outer (Lloyd) and 100 inner (barycenter)
iterations, as in Section 4.2. Training split only, capped at N_MAX series to keep
the cost reasonable. Evaluation: hard-DTW k-means energy (Eq. 5, gamma = 0).
Inits: k random series of the dataset (3 seeds) and Euclidean k-means centroids.

Usage:  python -m experiments.exp_kmeans [n_jobs]  ->  results/kmeans.csv
        (SOFTDTW_FULL=1: the paper's protocol on its 57 / 58 datasets; see exp_kmeans_subset.py)
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from tslearn.datasets import UCR_UEA_datasets

from softdtw.clustering import dtw_kmeans, kmeans_energy
from experiments.datasets import FULL, KMEANS_EUCLIDEAN, KMEANS_RANDOM, OUT, checkpointed

DATASETS = ["ItalyPowerDemand", "TwoLeadECG", "ECG200", "CBF", "ECGFiveDays",
            "GunPoint", "Plane", "ArrowHead", "Coffee", "Trace"]
GAMMAS = [1.0, 0.1, 0.01, 0.001]
N_MAX, SEEDS = 100, [0, 1, 2]
DATASETS_RANDOM = DATASETS_EUCLIDEAN = DATASETS
CONFIGS = [("soft-DTW", g) for g in GAMMAS] + [("DBA", None), ("subgradient", None)]

if FULL:  # paper: whole training split, one random init per dataset
    N_MAX, SEEDS = None, [0]
    DATASETS_RANDOM, DATASETS_EUCLIDEAN = KMEANS_RANDOM, KMEANS_EUCLIDEAN


def load_train(name):
    X, y, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    if N_MAX is not None and len(X) > N_MAX:
        idx = np.random.default_rng(0).choice(len(X), N_MAX, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float64), y


def euclidean_kmeans_init(X, k, seed):
    km = KMeans(k, n_init=10, random_state=seed).fit(X.reshape(len(X), -1))
    return km.cluster_centers_.reshape(k, *X.shape[1:])


def run(name, init_name, seed, method, gamma):
    """One (dataset, init, method) job: the init depends only on (dataset, init, seed)."""
    warnings.filterwarnings("ignore")
    X, y = load_train(name)
    k = len(np.unique(y))
    rng = np.random.default_rng(seed)

    init = (X[rng.choice(len(X), k, replace=False)] if init_name == "random" else euclidean_kmeans_init(X, k, seed))

    t0 = time.perf_counter()
    C, _ = dtw_kmeans(X, k, method, init, gamma=gamma)

    return [dict(dataset=name, init=init_name, seed=seed, method=method,
                 gamma=gamma, energy=kmeans_energy(X, C), time=time.perf_counter() - t0)]


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    size = {}

    for d in sorted(set(DATASETS_RANDOM) | set(DATASETS_EUCLIDEAN)):  # downloads each dataset once
        X, _ = load_train(d)
        size[d] = X.shape[0] * X.shape[1] ** 2

    jobs = [(d, "random", s) for d in DATASETS_RANDOM for s in SEEDS] + \
           [(d, "euclidean", 0) for d in DATASETS_EUCLIDEAN]
    jobs = [(d, i, s, m, g) for d, i, s in jobs for m, g in CONFIGS]
    jobs.sort(key=lambda j: -size[j[0]])  # longest first, for load balancing
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(checkpointed)(f"{OUT}/kmeans_parts/{d}_{i}_{s}_{m}_{g}.csv",
                              lambda j=(d, i, s, m, g): run(*j))
        for d, i, s, m, g in jobs)

    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/kmeans.csv", index=False)
    print(df.groupby(["init", "method", "gamma"], dropna=False)["energy"].mean())
