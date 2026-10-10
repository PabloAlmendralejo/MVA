"""Forecasting with soft-DTW_omega for every gamma of Table 3 (1, 0.1, 0.001;
gamma = 0.01 is in exp_forecast_omega.py), omega in {0.1, 1}, random init and
init from the Euclidean network. Training is deterministic (same seed), so the
results are directly comparable with results/forecast.csv.

Usage:  python exp_forecast_omega_gammas.py  ->  results/forecast_omega_gammas.csv
"""

import copy
import os
import warnings

import pandas as pd

from exp_forecast import DATASETS, euclidean_loss, evaluate, load, mlp, train
from exp_forecast_omega import timing_errors
from torch_sdtw import SoftDTWLoss

GAMMAS = [1.0, 0.1, 0.001]
OMEGAS = [0.1, 1.0]


def run(name):
    Xin, Xout, Tin, Tout = load(name)
    T = Tout.numpy()
    base = train(mlp(Xin.shape[1], Xout.shape[1]), euclidean_loss, Xin, Xout)
    rows = []
    for gamma in GAMMAS:
        for omega in OMEGAS:
            for init in ("random", "euclidean"):
                model = mlp(Xin.shape[1], Xout.shape[1]) if init == "random" else copy.deepcopy(base)
                model = train(model, SoftDTWLoss(gamma, omega=omega), Xin, Xout)
                e, d, P = evaluate(model, Tin, Tout)
                tmax, tmin = timing_errors(P, T)
                rows.append(dict(dataset=name, init=init, model="soft-DTW_omega", gamma=gamma,
                                 omega=omega, euclid=e, dtw=d, t_max_err=tmax, t_min_err=tmin))
    return rows


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs("results/forecast_omega_gammas_parts", exist_ok=True)
    for name in DATASETS:  # checkpointed per dataset
        part = f"results/forecast_omega_gammas_parts/{name}.csv"
        if os.path.exists(part):
            continue
        pd.DataFrame(run(name)).to_csv(part, index=False)
        print(name, "done", flush=True)
    pd.concat([pd.read_csv(f"results/forecast_omega_gammas_parts/{d}.csv") for d in DATASETS]).to_csv(
        "results/forecast_omega_gammas.csv", index=False)
