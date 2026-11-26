"""Nearest-centroid classification with soft-DTW_omega (extension of Fig. 7).

Same pooled data, splits and protocol as exp_classification.py: one barycenter
per class (soft-DTW_omega, initialised at the class Euclidean mean), test series
assigned to the closest centroid under soft-DTW_omega; gamma (15 log-spaced values
in [1e-3, 10]) and omega in {0.1, 1} are selected jointly on validation.

Usage:  python -m experiments.exp_classification_omega [n_jobs]  ->  results/classification_omega.csv

Full-paper mode (SOFTDTW_FULL=1): the 79 datasets and 3 splits of Fig. 7, with 5 of the
15 gammas, as far apart as possible (1e-3, 1.39e-2, 0.1, 0.72, 10), times omega in {0.1, 1}.
The pair with the best validation accuracy is kept (smallest gamma, then omega, on ties).
Soft-DTW on the same 5 gammas is in classification_all_gammas.csv.
Usage:  python -m experiments.exp_classification_omega [n_jobs]
        ->  results_full/classification_omega.csv (selected pair), classification_omega_all.csv
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one thread per joblib worker; before numba is imported

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from softdtw.omega import soft_barycenter, pairwise_adtw
from experiments.datasets import FULL, OUT, checkpointed
from experiments.exp_classification import DATASETS, GAMMAS, SPLITS, load_all, split

OMEGAS = [0.1, 1.0]
GAMMAS_FULL = GAMMAS[[0, 4, 7, 10, 14]]


def fit(X, y, gamma, omega, max_iter=100):
    classes = np.unique(y)
    C = [soft_barycenter(list(X[y == c]), "adtw", gamma, X[y == c].mean(0),
                           omega=omega, max_iter=max_iter) for c in classes]

    return classes, np.array(C).reshape(len(classes), *X.shape[1:])


def run(name, seed):
    part = f"results/classification_omega_parts/{name}_{seed}.csv"

    if os.path.exists(part):
        return pd.read_csv(part).to_dict("records")

    warnings.filterwarnings("ignore")
    X, y = load_all(name)
    tr, va, te = split(y, seed)
    acc = lambda pred, idx: float(np.mean(pred == y[idx]))
    best = (-1, None, None, None, None)

    for g in GAMMAS:
        for w in OMEGAS:
            classes, C = fit(X[tr], y[tr], g, w)
            a = acc(classes[pairwise_adtw(X[va], C, g, w).argmin(1)], va)
            if a > best[0]:
                best = (a, g, w, classes, C)

    _, g, w, classes, C = best
    rows = [dict(method="soft-DTW_omega", gamma=g, omega=w,
                 acc=acc(classes[pairwise_adtw(X[te], C, g, w).argmin(1)], te),
                 dataset=name, split=seed)]
    os.makedirs(os.path.dirname(part), exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def run_pair(name, seed, g, w):
    """Full mode: validation and test accuracy of soft-DTW_omega for one (gamma, omega)."""
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    X, y = load_all(name)
    tr, va, te = split(y, seed)
    classes, C = fit(X[tr], y[tr], g, w)
    acc = lambda idx: float(np.mean(classes[pairwise_adtw(X[idx], C, g, w).argmin(1)] == y[idx]))
    return [dict(method="soft-DTW_omega", gamma=g, omega=w, val_acc=acc(va), acc=acc(te),
                 dataset=name, split=seed, time=time.perf_counter() - t0)]


def select_pair(df):
    """Row of each (dataset, split) with the best validation accuracy
    (smallest gamma, then smallest omega, on ties)."""
    df = df.sort_values(["gamma", "omega"])
    return df.loc[df.groupby(["dataset", "split"]).val_acc.idxmax()]


if __name__ == "__main__" and FULL:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cost = {}

    for d in DATASETS:  # downloads each dataset once
        X, _ = load_all(d)
        cost[d] = X.shape[0] * X.shape[1] ** 2

    # same job order as exp_classification.py
    pairs = [(g, w) for g in GAMMAS_FULL for w in OMEGAS]
    ordered = sorted(DATASETS, key=lambda d: -cost[d])
    jobs = [(d, s, g, w) for s in SPLITS for i in range(0, len(ordered), 2)
            for g, w in pairs for d in ordered[i:i + 2]]

    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(checkpointed)(f"{OUT}/classification_omega_parts/{d}_{s}_gamma{g:.6g}_omega{w:g}.csv",
                              lambda d=d, s=s, g=g, w=w: run_pair(d, s, g, w))
        for d, s, g, w in jobs)

    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/classification_omega_all.csv", index=False)
    best = select_pair(df)
    best.to_csv(f"{OUT}/classification_omega.csv", index=False)
    print(best.groupby("dataset")[["acc", "gamma", "omega"]].mean())

elif __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run)(d, s) for d in DATASETS for s in SPLITS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/classification_omega.csv", index=False)
    print(df.groupby("dataset")[["acc", "gamma", "omega"]].mean())
