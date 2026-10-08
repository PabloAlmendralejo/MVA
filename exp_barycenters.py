"""Table 1 of the paper (reduced): soft-DTW barycenters vs DBA vs subgradient.

Protocol (Section 4.1): for each dataset, pick a class at random and 10 series of
that class, compute their barycenter with every method from the same init, and
evaluate the hard DTW loss (Eq. 4, gamma = 0). Repeat 10 times. Max 100 iterations.
Two inits: random N(0, 1) (UCR series are z-normalised) and the Euclidean mean.

Usage:  python exp_barycenters.py [n_jobs]   ->  results/barycenters.csv
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from barycenter import (dba_barycenter, dtw_barycenter_loss, euclidean_mean,
                        softdtw_barycenter, subgradient_barycenter)

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "MedicalImages", "CBF", "ECGFiveDays", "Plane", "GunPoint",
            "ArrowHead", "Trace", "Coffee"]
GAMMAS = [1.0, 0.1, 0.01, 0.001]
N_REPS, N_SERIES, MAX_ITER = 10, 10, 100


def load(name):
    from tslearn.datasets import UCR_UEA_datasets
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return np.concatenate([X, Xt]), np.concatenate([y, yt])


def run_dataset(name, seed=0):
    warnings.filterwarnings("ignore")
    X, y = load(name)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(N_REPS):
        cls = rng.choice(np.unique(y))
        members = np.where(y == cls)[0]
        idx = rng.choice(members, min(N_SERIES, len(members)), replace=False)
        Y = [X[i] for i in idx]
        inits = {"random": rng.normal(size=Y[0].shape), "euclidean": euclidean_mean(Y)}
        for init_name, init in inits.items():
            methods = [(f"soft-DTW", g, lambda g=g: softdtw_barycenter(Y, g, init, max_iter=MAX_ITER))
                       for g in GAMMAS]
            methods += [("DBA", np.nan, lambda: dba_barycenter(Y, init, max_iter=MAX_ITER)),
                        ("subgradient", np.nan, lambda: subgradient_barycenter(Y, init, max_iter=MAX_ITER))]
            for method, gamma, fn in methods:
                t0 = time.perf_counter()
                xb = fn()
                rows.append(dict(dataset=name, length=Y[0].shape[0], rep=rep, cls=cls,
                                 init=init_name, method=method, gamma=gamma,
                                 loss=dtw_barycenter_loss(xb, Y),
                                 time=time.perf_counter() - t0))
    return rows


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for name in DATASETS:  # download sequentially once
        load(name)
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_dataset)(d) for d in DATASETS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/barycenters.csv", index=False)
    print(df.groupby(["init", "method", "gamma"], dropna=False)["loss"].mean())
