"""Table 2 of the paper (reduced): k-means under soft-DTW vs DBA vs subgradient.

k = number of classes; at most 30 outer (Lloyd) and 100 inner (barycenter)
iterations, as in Section 4.2. Training split only, capped at N_MAX series to keep
the cost reasonable. Evaluation: hard-DTW k-means energy (Eq. 5, gamma = 0).
Inits: k random series of the dataset (3 seeds) and Euclidean k-means centroids.

Usage:  python exp_kmeans.py [n_jobs]  ->  results/kmeans.csv
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from clustering import dtw_kmeans, kmeans_energy

DATASETS = ["ItalyPowerDemand", "TwoLeadECG", "ECG200", "CBF", "ECGFiveDays",
            "GunPoint", "Plane", "ArrowHead", "Coffee", "Trace"]
GAMMAS = [1.0, 0.1, 0.01, 0.001]
N_MAX, SEEDS = 100, [0, 1, 2]


def load_train(name):
    from tslearn.datasets import UCR_UEA_datasets
    X, y, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    if len(X) > N_MAX:
        idx = np.random.default_rng(0).choice(len(X), N_MAX, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float64), y


def euclidean_kmeans_init(X, k, seed):
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=10, random_state=seed).fit(X.reshape(len(X), -1))
    return km.cluster_centers_.reshape(k, *X.shape[1:])


def run(name, init_name, seed):
    warnings.filterwarnings("ignore")
    X, y = load_train(name)
    k = len(np.unique(y))
    rng = np.random.default_rng(seed)
    init = (X[rng.choice(len(X), k, replace=False)] if init_name == "random"
            else euclidean_kmeans_init(X, k, seed))
    rows = []
    configs = [("soft-DTW", g) for g in GAMMAS] + [("DBA", None), ("subgradient", None)]
    for method, gamma in configs:
        t0 = time.perf_counter()
        C, labels = dtw_kmeans(X, k, method, init, gamma=gamma)
        rows.append(dict(dataset=name, init=init_name, seed=seed, method=method,
                         gamma=gamma, energy=kmeans_energy(X, C),
                         time=time.perf_counter() - t0))
    return rows


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(d, "random", s) for d in DATASETS for s in SEEDS] + \
           [(d, "euclidean", 0) for d in DATASETS]
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/kmeans.csv", index=False)
    print(df.groupby(["init", "method", "gamma"], dropna=False)["energy"].mean())
