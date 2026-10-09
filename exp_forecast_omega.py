"""Forecasting (Table 3 protocol) with the soft-DTW variants: does a warping
penalty prevent the degenerate soft-DTW forecasts (flat line + spikes)?

Same data, MLP and training as exp_forecast.py. Training losses: Euclidean,
soft-DTW, its divergence, and soft-DTW with a warping penalty omega, all with
gamma = 0.01; random init and init from the Euclidean network.
Test metrics: Euclidean and hard DTW (as in the paper) plus the timing error of
the maximum and of the minimum of the forecast (|argmax pred - argmax truth|, in
% of the horizon), which measures predictions that are right in shape but
wrong in time.

Usage:  python exp_forecast_omega.py  ->  results/forecast_omega.csv,
        results/forecast_omega_examples.npz
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd

from exp_forecast import DATASETS, euclidean_loss, evaluate, load, mlp, train
from torch_sdtw import SoftDTWLoss

GAMMA = 0.01
LOSSES = {  # label -> (omega, divergence)
    "soft-DTW": (0.0, False),
    "soft-DTW divergence": (0.0, True),
    "soft-DTW_omega 0.1": (0.1, False),
    "soft-DTW_omega 1": (1.0, False),
}


def timing_errors(P, T):
    h = T.shape[1]
    tmax = np.abs(P.argmax(1) - T.argmax(1)).mean() / h * 100
    tmin = np.abs(P.argmin(1) - T.argmin(1)).mean() / h * 100
    return float(tmax), float(tmin)


def run(name):
    Xin, Xout, Tin, Tout = load(name)
    T = Tout.numpy()
    rows, preds = [], {}

    def record(label, init, model):
        e, d, P = evaluate(model, Tin, Tout)
        tmax, tmin = timing_errors(P, T)
        rows.append(dict(dataset=name, init=init, model=label, euclid=e, dtw=d,
                         t_max_err=tmax, t_min_err=tmin))
        return P

    base = train(mlp(Xin.shape[1], Xout.shape[1]), euclidean_loss, Xin, Xout)
    preds["Euclidean"] = record("Euclidean", "random", base)
    record("Euclidean", "euclidean", base)
    for label, (omega, div) in LOSSES.items():
        for init in ("random", "euclidean"):
            model = mlp(Xin.shape[1], Xout.shape[1]) if init == "random" else copy.deepcopy(base)
            model = train(model, SoftDTWLoss(GAMMA, omega=omega, divergence=div), Xin, Xout)
            P = record(label, init, model)
            if init == "random":
                preds[label] = P
    return rows, preds, Tin.numpy(), T


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs("results/forecast_omega_parts", exist_ok=True)
    for name in DATASETS:  # checkpointed per dataset
        part = f"results/forecast_omega_parts/{name}"
        if os.path.exists(part + ".csv"):
            continue
        rows, preds, Tin, Tout = run(name)
        k = slice(0, 5)
        np.savez(part + ".npz", input=Tin[k], truth=Tout[k], **{m: P[k] for m, P in preds.items()})
        pd.DataFrame(rows).to_csv(part + ".csv", index=False)
        print(name, "done", flush=True)
    pd.concat([pd.read_csv(f"results/forecast_omega_parts/{d}.csv") for d in DATASETS]).to_csv(
        "results/forecast_omega.csv", index=False)
    np.savez("results/forecast_omega_examples.npz",
             **{f"{d}/{k}": v for d in DATASETS for k, v in np.load(f"results/forecast_omega_parts/{d}.npz").items()})
