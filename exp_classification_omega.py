"""Nearest-centroid classification with soft-DTW_omega (extension of Fig. 7).

Same pooled data, splits and protocol as exp_classification.py: one barycenter
per class (soft-DTW_omega, initialised at the class Euclidean mean), test series
assigned to the closest centroid under soft-DTW_omega; gamma (15 log-spaced values
in [1e-3, 10]) and omega in {0.1, 1} are selected jointly on validation.

Usage:  python exp_classification_omega.py [n_jobs]  ->  results/classification_omega.csv
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one thread per joblib worker

import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from exp_classification import DATASETS, GAMMAS, SPLITS, load_all, split
from omega import soft_barycenter, pairwise_adtw

OMEGAS = [0.1, 1.0]


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


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run)(d, s) for d in DATASETS for s in SPLITS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/classification_omega.csv", index=False)
    print(df.groupby("dataset")[["acc", "gamma", "omega"]].mean())
