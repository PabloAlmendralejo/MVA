"""Figure 7 of the paper: nearest-centroid classification, soft-DTW vs DBA.

Protocol (Section 4.3): pool train + test, split 50% train / 25% validation /
25% test (stratified). For soft-DTW, gamma is chosen on validation among 15
log-spaced values in [1e-3, 10]; centroids are soft-DTW barycenters and prediction
uses dtw_gamma. DBA centroids are predicted with hard DTW. Extra baselines the
paper does not report: Euclidean mean + Euclidean distance, and 1-NN DTW (reduced
mode only). Repeated over 3 random splits.

Usage:  python -m experiments.exp_classification [n_jobs]  ->  results/classification.csv
        and classification_all_gammas.csv (validation and test accuracy of every gamma)
        (SOFTDTW_FULL=1: the 79 datasets, in results_full/)
"""

import os
import sys
import time
import warnings

import numba
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tslearn.datasets import UCR_UEA_datasets

from softdtw.clustering import nearest_centroid_fit, nearest_centroid_predict, pairwise_sdtw
from experiments.datasets import CLASSIFICATION, FULL, OUT, checkpointed

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "CBF", "ECGFiveDays", "GunPoint", "Plane", "ArrowHead", "Coffee",
            "Trace", "MedicalImages"]
if FULL:
    DATASETS = CLASSIFICATION
GAMMAS = np.logspace(-3, 1, 15)
SPLITS = [0, 1, 2]


def load_all(name):
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return np.concatenate([X, Xt]).astype(np.float64), np.concatenate([y, yt])


def split(y, seed):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=0.5, stratify=y, random_state=seed)
    try:
        va, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    except ValueError:  # a class left with one member (Phoneme) cannot be stratified
        va, te = train_test_split(rest, train_size=0.5, random_state=seed)
    return tr, va, te


def run_gamma(name, seed, g):
    """soft-DTW nearest centroid for one gamma: validation and test accuracy."""
    X, y = load_all(name)
    tr, va, te = split(y, seed)
    classes, C = nearest_centroid_fit(X[tr], y[tr], "soft-DTW", gamma=g)
    acc = lambda idx: float(np.mean(nearest_centroid_predict(X[idx], classes, C, g) == y[idx]))
    return [dict(method="soft-DTW", gamma=g, val_acc=acc(va), acc=acc(te), dataset=name, split=seed)]


def run_baselines(name, seed):
    """DBA (predicted with hard DTW), plus baselines the paper does not report:
    Euclidean NC and, in reduced mode only, 1-NN DTW (too slow on the full archive)."""
    X, y = load_all(name)
    tr, va, te = split(y, seed)
    acc = lambda pred: float(np.mean(pred == y[te]))
    rows = []
    classes, C = nearest_centroid_fit(X[tr], y[tr], "DBA")
    rows.append(dict(method="DBA", gamma=np.nan, acc=acc(nearest_centroid_predict(X[te], classes, C, 0.0))))
    classes, C = nearest_centroid_fit(X[tr], y[tr], "euclidean")
    d = ((X[te][:, None] - C[None]) ** 2).sum((2, 3))
    rows.append(dict(method="Euclidean NC", gamma=np.nan, acc=acc(classes[d.argmin(1)])))
    if not FULL:
        D = pairwise_sdtw(X[te], X[tr], 0.0)
        rows.append(dict(method="1-NN DTW", gamma=np.nan, acc=acc(y[tr][D.argmin(1)])))
    for r in rows:
        r.update(dataset=name, split=seed)
    return rows


def job(kind, name, seed, g, n_threads):
    """One checkpointed job: ('gamma', dataset, split, gamma) or ('baselines', dataset, split)."""
    warnings.filterwarnings("ignore")
    numba.set_num_threads(n_threads)

    def timed(fn):  # job run time, shared among its rows
        t0 = time.perf_counter()
        rows = fn()
        for r in rows:
            r["time"] = (time.perf_counter() - t0) / len(rows)
        return rows

    if kind == "gamma":
        return checkpointed(f"{OUT}/classification_parts/{name}_{seed}_gamma{g:.6g}.csv",
                            lambda: timed(lambda: run_gamma(name, seed, g)))
    return checkpointed(f"{OUT}/classification_parts/{name}_{seed}_baselines.csv",
                        lambda: timed(lambda: run_baselines(name, seed)))


def select_gamma(df):
    """soft-DTW row of each (dataset, split): the gamma with the best validation accuracy
    (the smallest gamma on ties), with its test accuracy."""
    soft = df[df.method == "soft-DTW"].sort_values("gamma")
    best = soft.loc[soft.groupby(["dataset", "split"]).val_acc.idxmax()]
    return pd.concat([best, df[df.method != "soft-DTW"]])[["method", "gamma", "acc", "dataset", "split"]]


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_threads = 1 if n_jobs > 1 else (os.cpu_count() or 1)
    cost = {}
    for d in DATASETS:  # downloads each dataset once
        X, _ = load_all(d)
        cost[d] = X.shape[0] * X.shape[1] ** 2
    # One job per (dataset, split, gamma) plus one for the baselines. Split by split,
    # longest series first, with two datasets interleaved so that the memory-heavy jobs
    # of the longest series do not all run at once.
    kinds = [("gamma", g) for g in GAMMAS] + [("baselines", np.nan)]
    ordered = sorted(DATASETS, key=lambda d: -cost[d])
    jobs = [(kind, d, s, g) for s in SPLITS for i in range(0, len(ordered), 2)
            for kind, g in kinds for d in ordered[i:i + 2]]
    out = Parallel(n_jobs=n_jobs, verbose=10)(delayed(job)(*j, n_threads) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/classification_all_gammas.csv", index=False)
    df = select_gamma(df)
    df.to_csv(f"{OUT}/classification.csv", index=False)
    print(df.groupby(["dataset", "method"])["acc"].mean().unstack())
