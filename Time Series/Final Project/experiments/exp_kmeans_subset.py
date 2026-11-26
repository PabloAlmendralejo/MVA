"""k-means (Table 2 of the paper) on the 41 cheapest of its 57 datasets, with every method:
soft-DTW (4 gammas), DBA, subgradient and soft-DTW_omega (4 gammas x omega in {0.1, 1}).

Protocol of exp_kmeans.py with SOFTDTW_FULL=1: whole training split, one random init
(k random series) and one Euclidean k-means init, shared by all methods. Score: hard-DTW
k-means energy (Eq. 5), as in the paper, plus ARI / NMI against the true classes, which do
not favour the methods that optimise hard DTW.

The 16 datasets left out (long series or thousands of training series) would take ~96% of
the CPU time. Soft-DTW, DBA and subgradient jobs are checkpointed under the file names of
exp_kmeans.py, so a full Table 2 run reuses them.

Usage:  python -m experiments.exp_kmeans_subset [n_jobs]  ->  results_full/kmeans_subset.csv
"""

import os

os.environ.setdefault("SOFTDTW_FULL", "1")       # paper protocol; before experiments.* is imported
os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # one thread per joblib worker; before numba is imported

import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from softdtw.clustering import dtw_kmeans, kmeans_energy
from experiments.datasets import KMEANS_RANDOM, OUT, checkpointed
from experiments.exp_kmeans import GAMMAS, euclidean_kmeans_init, load_train

DATASETS = [
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "MoteStrain", "TwoLeadECG", "ECGFiveDays",
    "CBF", "GunPoint", "ECG200", "DiatomSizeReduction", "SyntheticControl", "Coffee", "ArrowHead",
    "FaceFour", "Wine", "ShapeletSim", "BeetleFly", "BirdChicken", "MiddlePhalanxTW",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "ProximalPhalanxTW",
    "MiddlePhalanxOutlineAgeGroup", "FacesUCR", "Beef", "MedicalImages",
    "MiddlePhalanxOutlineCorrect", "DistalPhalanxOutlineCorrect", "Trace", "ECG5000", "FaceAll",
    "Car", "Adiac", "ChlorineConcentration", "Ham", "WordSynonyms", "TwoPatterns", "Fish",
    "FiftyWords", "CricketX", "CricketZ", "CricketY",
]
assert set(DATASETS) <= set(KMEANS_RANDOM) and len(set(DATASETS)) == len(DATASETS) == 41
INITS = [("random", 0), ("euclidean", 0)]
OMEGAS = [0.1, 1.0]
CONFIGS = ([("soft-DTW", g, 0.0) for g in GAMMAS] + [("DBA", None, 0.0), ("subgradient", None, 0.0)]
           + [("soft-DTW_omega", g, w) for g in GAMMAS for w in OMEGAS])
# relative run time of each config, measured on the reduced run (soft-DTW_omega: by omega)
RATE = {("soft-DTW", 1.0): 2.0, ("soft-DTW", 0.1): 3.1, ("soft-DTW", 0.01): 4.7,
        ("soft-DTW", 0.001): 11.0, ("DBA", None): 0.9, ("subgradient", None): 2.5}


def part(d, init, seed, method, gamma, omega):
    if method == "soft-DTW_omega":
        return f"{OUT}/kmeans_omega_parts/{d}_{init}_{seed}_gamma{gamma:g}_omega{omega:g}.csv"
    return f"{OUT}/kmeans_parts/{d}_{init}_{seed}_{method}_{gamma}.csv"


def run(name, init_name, seed, method, gamma, omega):
    """One (dataset, init, method) job; the init depends only on (dataset, init, seed)."""
    warnings.filterwarnings("ignore")
    X, y = load_train(name)
    k = len(np.unique(y))
    rng = np.random.default_rng(seed)
    init = (X[rng.choice(len(X), k, replace=False)] if init_name == "random"
            else euclidean_kmeans_init(X, k, seed))
    t0 = time.perf_counter()
    C, labels = dtw_kmeans(X, k, method, init, gamma=gamma, omega=omega)
    elapsed = time.perf_counter() - t0
    return [dict(dataset=name, init=init_name, seed=seed, method=method, gamma=gamma,
                 omega=omega if method == "soft-DTW_omega" else np.nan,
                 energy=kmeans_energy(X, C), ari=adjusted_rand_score(y, labels),
                 nmi=normalized_mutual_info_score(y, labels), time=elapsed)]


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    size = {}
    for d in DATASETS:  # downloads each dataset once; cost ~ n_train^1.3 * length^2
        X, _ = load_train(d)
        size[d] = X.shape[0] ** 1.3 * X.shape[1] ** 2
    jobs = [(d, i, s, m, g, w) for d in DATASETS for i, s in INITS for m, g, w in CONFIGS]
    jobs.sort(key=lambda j: -size[j[0]] * RATE.get((j[3], j[4]), 3.0 if j[5] == 0.1 else 2.2))
    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(checkpointed)(part(*j), lambda j=j: run(*j)) for j in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/kmeans_subset.csv", index=False)
    print(df.groupby(["init", "method", "gamma", "omega"], dropna=False)[["energy", "ari"]].mean())
