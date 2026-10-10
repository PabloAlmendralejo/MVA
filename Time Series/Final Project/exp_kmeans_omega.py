"""k-means with soft-DTW_omega (extension of Table 2 of the paper).

Same datasets, inits (k random series, 3 seeds, and Euclidean k-means), iteration
limits and hard-DTW energy as exp_kmeans.py; assignment and centring both use
soft-DTW_omega. Checkpointed per job in results/kmeans_omega_parts/.

Usage:  python exp_kmeans_omega.py [n_jobs]  ->  results/kmeans_omega.csv
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one thread per joblib worker

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from clustering import dtw_kmeans, kmeans_energy
from exp_kmeans import DATASETS, GAMMAS, SEEDS, euclidean_kmeans_init, load_train

OMEGAS = [0.1, 1.0]


def run(name, init_name, seed):
    part = f"results/kmeans_omega_parts/{name}_{init_name}_{seed}.csv"
    if os.path.exists(part):
        return pd.read_csv(part).to_dict("records")
    warnings.filterwarnings("ignore")
    X, y = load_train(name)
    k = len(np.unique(y))
    rng = np.random.default_rng(seed)  # same initial centroids as exp_kmeans.py
    init = (X[rng.choice(len(X), k, replace=False)] if init_name == "random"
            else euclidean_kmeans_init(X, k, seed))
    rows = []
    for gamma in GAMMAS:
        for omega in OMEGAS:
            t0 = time.perf_counter()
            C, _ = dtw_kmeans(X, k, "soft-DTW_omega", init, gamma=gamma, omega=omega)
            rows.append(dict(dataset=name, init=init_name, seed=seed, method="soft-DTW_omega",
                             gamma=gamma, omega=omega, energy=kmeans_energy(X, C),
                             time=time.perf_counter() - t0))
    os.makedirs(os.path.dirname(part), exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    jobs = [(d, "random", s) for d in DATASETS for s in SEEDS] + \
           [(d, "euclidean", 0) for d in DATASETS]
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/kmeans_omega.csv", index=False)
    print(df.groupby(["init", "gamma", "omega"])["energy"].mean())
