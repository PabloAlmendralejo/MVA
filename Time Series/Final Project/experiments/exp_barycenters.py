"""Table 1 of the paper: soft-DTW barycenters vs DBA vs subgradient.

Protocol (Section 4.1): for each dataset, pick a class at random and 10 series of
that class, compute their barycenter with every method from the same init, and
evaluate the hard DTW loss (Eq. 4, gamma = 0). Repeat 10 times. Max 100 iterations.
Two inits: random N(0, 1) (UCR series are z-normalised) and the Euclidean mean.

Usage:  python -m experiments.exp_barycenters [n_jobs]   ->  results/barycenters.csv
        (SOFTDTW_FULL=1: the 79 datasets of the paper, in results_full/)
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tslearn.datasets import UCR_UEA_datasets

from softdtw.barycenter import (dba_barycenter, dtw_barycenter_loss, euclidean_mean,
                                softdtw_barycenter, subgradient_barycenter)
from experiments.datasets import BARYCENTERS, FULL, OUT, checkpointed

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "MedicalImages", "CBF", "ECGFiveDays", "Plane", "GunPoint",
            "ArrowHead", "Trace", "Coffee"]
if FULL:
    DATASETS = BARYCENTERS
GAMMAS = [1.0, 0.1, 0.01, 0.001]
N_REPS, N_SERIES, MAX_ITER = 10, 10, 100


def load(name):
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return np.concatenate([X, Xt]), np.concatenate([y, yt])


def run_dataset(name, seed=0, reps=None):
    """All repetitions, or only those in `reps`. Skipped repetitions still draw their
    random numbers, so each repetition always gets the same data and init."""
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
        if reps is not None and rep not in reps:
            continue
        for init_name, init in inits.items():
            methods = [("soft-DTW", g, lambda g=g: softdtw_barycenter(Y, g, init, max_iter=MAX_ITER))
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
    length = {d: load(d)[0].shape[1] for d in DATASETS}  # downloads each dataset once
    jobs = sorted(((d, r) for d in DATASETS for r in range(N_REPS)), key=lambda j: -length[j[0]])
    out = Parallel(n_jobs=n_jobs, verbose=10)(  # one job per repetition, longest series first
        delayed(checkpointed)(f"{OUT}/barycenters_parts/{d}_{r}.csv",
                              lambda d=d, r=r: run_dataset(d, reps=[r]))
        for d, r in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/barycenters.csv", index=False)
    print(df.groupby(["init", "method", "gamma"], dropna=False)["loss"].mean())
