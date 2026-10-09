"""Which soft surrogate gives the best DTW barycenter? (Table 1 protocol, more rows.)

The paper uses soft-DTW as a smooth surrogate to minimise the hard DTW barycenter
loss and compares it with DBA *on that loss*. We apply the same test to every
variant of soft-DTW we implement: its debiased divergence and soft-DTW with a
warping penalty omega (kind 'adtw' in the code). All are scored with the hard DTW loss (Eq. 4, gamma = 0):
  * train loss: on the 10 series the barycenter was computed from (as the paper);
  * held-out loss: on up to 20 other series of the same class (detects
    barycenters that fit the idiosyncrasies of the 10 series).
Draws (class, series, initialisations) are identical to exp_barycenters.py.

Usage:  python exp_family_barycenters.py [n_jobs]  ->  results/family_barycenters.csv
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one thread per joblib worker

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from barycenter import dba_barycenter, dtw_barycenter_loss, euclidean_mean
from exp_barycenters import DATASETS, N_REPS, N_SERIES, MAX_ITER, load
from family import family_barycenter

N_HELDOUT = 20
# (label, kind, gamma, omega, divergence)
CONFIGS = [
    ("soft-DTW", "dtw", 0.1, 0.0, False),
    ("soft-DTW", "dtw", 0.01, 0.0, False),
    ("soft-DTW divergence", "dtw", 0.1, 0.0, True),
    ("soft-DTW divergence", "dtw", 0.01, 0.0, True),
    ("soft-DTW_omega", "adtw", 0.1, 0.1, False),
    ("soft-DTW_omega", "adtw", 0.1, 1.0, False),
    ("soft-DTW_omega", "adtw", 0.01, 0.1, False),
    ("soft-DTW_omega", "adtw", 0.01, 1.0, False),
]


def run_dataset(name, seed=0):
    part = f"results/family_barycenters_parts/{name}.csv"
    if os.path.exists(part):
        return pd.read_csv(part).to_dict("records")
    warnings.filterwarnings("ignore")
    X, y = load(name)
    rng = np.random.default_rng(seed)  # same stream as exp_barycenters.py
    rows = []
    for rep in range(N_REPS):
        cls = rng.choice(np.unique(y))
        members = np.where(y == cls)[0]
        idx = rng.choice(members, min(N_SERIES, len(members)), replace=False)
        Y = [X[i] for i in idx]
        inits = {"random": rng.normal(size=Y[0].shape), "euclidean": euclidean_mean(Y)}
        rest = np.setdiff1d(members, idx)
        held = np.random.default_rng(1000 + rep).choice(rest, min(N_HELDOUT, len(rest)), replace=False)
        Yh = [X[i] for i in held]
        for init_name, init in inits.items():
            runs = [("DBA", None, np.nan, np.nan, False,
                     lambda: dba_barycenter(Y, init, max_iter=MAX_ITER))]
            runs += [(label, kind, gamma, omega, div,
                      lambda kind=kind, gamma=gamma, omega=omega, div=div: family_barycenter(
                          Y, kind, gamma, init, omega=omega, divergence=div, max_iter=MAX_ITER))
                     for label, kind, gamma, omega, div in CONFIGS]
            for label, kind, gamma, omega, div, fn in runs:
                t0 = time.perf_counter()
                xb = fn()
                rows.append(dict(dataset=name, rep=rep, init=init_name, method=label,
                                 gamma=gamma, omega=omega if kind == "adtw" else np.nan,
                                 divergence=div,
                                 train_loss=dtw_barycenter_loss(xb, Y),
                                 heldout_loss=dtw_barycenter_loss(xb, Yh) if Yh else np.nan,
                                 time=time.perf_counter() - t0))
    os.makedirs(os.path.dirname(part), exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_dataset)(d) for d in DATASETS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/family_barycenters.csv", index=False)
    print(df.groupby(["init", "method", "gamma", "omega"], dropna=False)[["train_loss", "heldout_loss"]].mean())
