"""Comparison of the soft-distance family with a neutral metric: 1-NN accuracy.

For every dataset and 3 stratified random splits (50% train / 25% validation /
25% test), each distance gets its hyper-parameters (gamma, omega, g) selected on
the validation set (1-NN against the training set), then is evaluated on test.
gamma = 0 (hard version) is part of the grid, so the selection says whether
smoothing helps. Each soft distance is tried raw and debiased (divergence).
Controls: Euclidean (no warping) and 1D Wasserstein on the value distribution
(no notion of time at all).

Usage:  python exp_family.py  ->  results/family.csv
"""

import itertools
import warnings

import numpy as np
import pandas as pd

from family import pairwise

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "CBF", "ECGFiveDays", "GunPoint", "Plane", "ArrowHead", "Coffee",
            "Trace", "MedicalImages"]
SPLITS = [0, 1, 2]
GAMMAS = [0.0, 0.01, 0.1, 1.0]
GRIDS = {
    "euclidean": [{}],
    "wasserstein": [{}],
    "dtw": [{"gamma": gm} for gm in GAMMAS],
    "adtw": [{"gamma": gm, "omega": om} for gm, om in
             itertools.product(GAMMAS, [0.03, 0.1, 0.3, 1.0, 3.0])],
    "wdtw": [{"gamma": gm, "g": g} for gm, g in itertools.product(GAMMAS, [0.01, 0.05, 0.25])],
    "erp": [{"gamma": gm} for gm in GAMMAS],
}


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


def one_nn_acc(X, y, ref, query, kind, divergence, params):
    D = pairwise(X[query], X[ref], kind, divergence=divergence, **params)
    return float(np.mean(y[ref][D.argmin(1)] == y[query]))


def run(name):
    X, y = load_all(name)
    rows = []
    for seed in SPLITS:
        tr, va, te = split(y, seed)
        for kind, grid in GRIDS.items():
            for divergence in ((False, True) if kind in ("dtw", "adtw", "wdtw", "erp") else (False,)):
                val = [one_nn_acc(X, y, tr, va, kind, divergence, p) for p in grid]
                best = grid[int(np.argmax(val))]  # ties -> first (smallest gamma / penalty)
                rows.append(dict(dataset=name, split=seed, kind=kind, divergence=divergence,
                                 val_acc=max(val),
                                 test_acc=one_nn_acc(X, y, tr, te, kind, divergence, best),
                                 **{f"best_{k}": v for k, v in best.items()}))
    return rows


if __name__ == "__main__":
    import os
    warnings.filterwarnings("ignore")
    os.makedirs("results/family_parts", exist_ok=True)
    for name in DATASETS:  # each dataset is checkpointed; finished ones are skipped
        part = f"results/family_parts/{name}.csv"
        if not os.path.exists(part):
            pd.DataFrame(run(name)).to_csv(part, index=False)
            print(name, "done", flush=True)
    pd.concat([pd.read_csv(f"results/family_parts/{d}.csv") for d in DATASETS]).to_csv(
        "results/family.csv", index=False)
