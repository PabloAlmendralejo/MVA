"""Data diagnosis for the Data section: size, normalisation, and how much
temporal misalignment there is within each class (the property DTW exploits).

For random pairs of series of the same class we measure
  * lag: the shift maximising their cross-correlation, in % of the length;
  * gain: 1 - DTW / Euclidean (squared costs), the share of the Euclidean cost
    removed by allowing time warping.
Usage:  python data_diagnosis.py  ->  results/data_diagnosis.csv
"""

import warnings

import numpy as np
import pandas as pd

from softdtw import dtw

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "MedicalImages", "CBF", "ECGFiveDays", "Plane", "GunPoint",
            "ArrowHead", "Trace", "Coffee", "ShapesAll"]
N_PAIRS = 200


def load_all(name):
    from tslearn.datasets import UCR_UEA_datasets
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return (np.concatenate([X, Xt])[..., 0].astype(np.float64),
            np.concatenate([y, yt]), len(X), len(Xt))


def best_lag(a, b):
    a, b = a - a.mean(), b - b.mean()
    c = np.correlate(a, b, mode="full")
    return abs(int(np.argmax(c)) - (len(b) - 1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    rows = []
    for name in DATASETS:
        X, y, ntr, nte = load_all(name)
        n = X.shape[1]
        lags, gains = [], []
        classes = np.unique(y)
        for _ in range(N_PAIRS):
            c = rng.choice(classes)
            members = np.where(y == c)[0]
            if len(members) < 2:
                continue
            i, j = rng.choice(members, 2, replace=False)
            lags.append(100 * best_lag(X[i], X[j]) / n)
            eu = ((X[i] - X[j]) ** 2).sum()
            gains.append(1 - dtw(X[i], X[j]) / eu)
        rows.append(dict(dataset=name, n_train=ntr, n_test=nte, length=n,
                         n_classes=len(classes),
                         mean_abs_mean=float(np.abs(X.mean(1)).mean()),
                         mean_std=float(X.std(1).mean()),
                         lag_median_pct=float(np.median(lags)),
                         lag_p90_pct=float(np.percentile(lags, 90)),
                         dtw_gain_median=float(np.median(gains))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/data_diagnosis.csv", index=False)
