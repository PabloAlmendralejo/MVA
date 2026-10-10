"""Figure 7 of the paper (reduced): nearest-centroid classification, soft-DTW vs DBA.

Protocol (Section 4.3): pool train + test, split 50% train / 25% validation /
25% test (stratified). For soft-DTW, gamma is chosen on validation among 15
log-spaced values in [1e-3, 10]; centroids are soft-DTW barycenters and prediction
uses dtw_gamma. DBA centroids are predicted with hard DTW. Extra baselines the
paper does not report: Euclidean mean + Euclidean distance, and 1-NN DTW.
Repeated over 3 random splits.

Usage:  python -m experiments.exp_classification [n_jobs]  ->  results/classification.csv
"""

import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from softdtw.clustering import nearest_centroid_fit, nearest_centroid_predict, pairwise_sdtw

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "CBF", "ECGFiveDays", "GunPoint", "Plane", "ArrowHead", "Coffee",
            "Trace", "MedicalImages"]
GAMMAS = np.logspace(-3, 1, 15)
SPLITS = [0, 1, 2]


def load_all(name):
    from tslearn.datasets import UCR_UEA_datasets
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return np.concatenate([X, Xt]).astype(np.float64), np.concatenate([y, yt])


def split(y, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=0.5, stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    return tr, va, te


def run(name, seed):
    """One (dataset, split) job; its rows are checkpointed to results/classification_parts/."""
    import os
    part = f"results/classification_parts/{name}_{seed}.csv"
    if os.path.exists(part):  # already done in a previous (interrupted) run
        return pd.read_csv(part).to_dict("records")
    rows = _run(name, seed)
    os.makedirs(os.path.dirname(part), exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def _run(name, seed):
    warnings.filterwarnings("ignore")
    X, y = load_all(name)
    tr, va, te = split(y, seed)
    acc = lambda pred, idx: float(np.mean(pred == y[idx]))
    rows = []

    # soft-DTW: select gamma on validation, refit is not needed (same train set)
    best = (-1, None, None, None)
    for g in GAMMAS:
        classes, C = nearest_centroid_fit(X[tr], y[tr], "soft-DTW", gamma=g)
        a = acc(nearest_centroid_predict(X[va], classes, C, g), va)
        if a > best[0]:
            best = (a, g, classes, C)
    _, g, classes, C = best
    rows.append(dict(method="soft-DTW", gamma=g,
                     acc=acc(nearest_centroid_predict(X[te], classes, C, g), te)))

    classes, C = nearest_centroid_fit(X[tr], y[tr], "DBA")
    rows.append(dict(method="DBA", gamma=np.nan,
                     acc=acc(nearest_centroid_predict(X[te], classes, C, 0.0), te)))

    classes, C = nearest_centroid_fit(X[tr], y[tr], "euclidean")
    d = ((X[te][:, None] - C[None]) ** 2).sum((2, 3))
    rows.append(dict(method="Euclidean NC", gamma=np.nan, acc=acc(classes[d.argmin(1)], te)))

    D = pairwise_sdtw(X[te], X[tr], 0.0)
    rows.append(dict(method="1-NN DTW", gamma=np.nan, acc=acc(y[tr][D.argmin(1)], te)))

    for r in rows:
        r.update(dataset=name, split=seed)
    return rows


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run)(d, s) for d in DATASETS for s in SPLITS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("results/classification.csv", index=False)
    print(df.groupby(["dataset", "method"])["acc"].mean().unstack())
