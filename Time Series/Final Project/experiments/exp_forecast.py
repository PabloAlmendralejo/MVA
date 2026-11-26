"""Table 3 / Figure 1 of the paper: multistep-ahead prediction with an MLP.

Protocol (Section 4.4): UCR train/test splits; the first 60% of each series is the
input, the remaining 40% the output (class labels ignored). One-hidden-layer MLP
with sigmoid activation, trained with Adam, under the Euclidean loss or the
soft-DTW loss (gamma in {1, 0.1, 0.01, 0.001}). Two initialisations of the
soft-DTW models: random, or the weights of the Euclidean-trained MLP.
Test metrics: Euclidean (sum of squares per series) and hard DTW; for each
dataset and metric the 5 models are ranked, then ranks are averaged.

Usage:  python -m experiments.exp_forecast [n_jobs]  ->  results/forecast.csv, results/forecast_examples.npz
        (SOFTDTW_FULL=1: the 76 datasets of the paper, in results_full/)
"""

import copy
import os
import sys
import warnings

import numba
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tslearn.datasets import UCR_UEA_datasets

from softdtw.clustering import pairwise_sdtw
from softdtw.torch_sdtw import SoftDTWLoss
from experiments.datasets import FORECAST, FULL, OUT

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "CBF", "ECGFiveDays", "GunPoint", "Plane", "ArrowHead", "Coffee",
            "Trace", "MedicalImages", "ShapesAll"]
if FULL:
    DATASETS = FORECAST
GAMMAS = [1.0, 0.1, 0.01, 0.001]
HIDDEN, EPOCHS, LR, SEED = 256, 400, 1e-3, 0
MAX_WORKERS = 6  # parallel datasets, to fit in 16 GB of RAM
torch.set_num_threads(4)


def load(name):
    X, _, Xt, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    X, Xt = X[..., 0].astype(np.float64), Xt[..., 0].astype(np.float64)
    t = int(round(0.6 * X.shape[1]))
    return (torch.tensor(X[:, :t]), torch.tensor(X[:, t:]),
            torch.tensor(Xt[:, :t]), torch.tensor(Xt[:, t:]))


def mlp(d_in, d_out):
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(d_in, HIDDEN), torch.nn.Sigmoid(),
                               torch.nn.Linear(HIDDEN, d_out)).double()


def train(model, loss_fn, Xin, Xout):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):  # full batch
        opt.zero_grad()
        loss = loss_fn(model(Xin), Xout)
        loss.backward()
        opt.step()
    return model


def euclidean_loss(pred, target):
    return ((pred - target) ** 2).sum(1).mean()


def evaluate(model, Xin, Xout):
    with torch.no_grad():
        P = model(Xin).numpy()
    T = Xout.numpy()
    eucl = float(((P - T) ** 2).sum(1).mean())
    dtw = float(np.mean([pairwise_sdtw(P[i:i + 1], T[i:i + 1], 0.0)[0, 0] for i in range(len(P))]))
    return eucl, dtw, P


def run(name):
    Xin, Xout, Tin, Tout = load(name)
    rows, preds = [], {}
    base = train(mlp(Xin.shape[1], Xout.shape[1]), euclidean_loss, Xin, Xout)
    e, d, P = evaluate(base, Tin, Tout)
    for init in ("random", "euclidean"):
        rows.append(dict(dataset=name, init=init, model="Euclidean", gamma=np.nan, euclid=e, dtw=d))
    preds["Euclidean"] = P
    for g in GAMMAS:
        for init in ("random", "euclidean"):
            model = mlp(Xin.shape[1], Xout.shape[1]) if init == "random" else copy.deepcopy(base)
            model = train(model, SoftDTWLoss(g), Xin, Xout)
            e, d, P = evaluate(model, Tin, Tout)
            rows.append(dict(dataset=name, init=init, model="soft-DTW", gamma=g, euclid=e, dtw=d))
            if init == "random":
                preds[f"soft-DTW {g}"] = P
    return rows, preds, Tin.numpy(), Tout.numpy()


def run_and_save(name, n_threads, run_fn, folder):
    """One dataset in a worker process; its csv, written last and atomically, marks it done."""
    warnings.filterwarnings("ignore")
    torch.set_num_threads(n_threads)
    numba.set_num_threads(n_threads)
    part = f"{OUT}/{folder}/{name}"
    if os.path.exists(part + ".csv"):  # finished in a previous run
        return
    rows, preds, Tin, Tout = run_fn(name)
    k = slice(0, 5)  # keep a few test series for the figures
    np.savez(part + ".tmp.npz", input=Tin[k], truth=Tout[k], **{m: P[k] for m, P in preds.items()})
    os.replace(part + ".tmp.npz", part + ".npz")
    pd.DataFrame(rows).to_csv(part + ".tmp", index=False)
    os.replace(part + ".tmp", part + ".csv")
    print(name, "done", flush=True)


def run_all(n_jobs, run_fn, name):
    """Every dataset, checkpointed in OUT/<name>_parts/  ->  OUT/<name>.csv, OUT/<name>_examples.npz"""
    folder = f"{name}_parts"
    os.makedirs(f"{OUT}/{folder}", exist_ok=True)
    cost = {}
    for d in DATASETS:  # downloads each dataset once
        Xin, Xout, _, _ = load(d)
        cost[d] = Xin.shape[0] * Xout.shape[1] ** 2
    order = sorted(DATASETS, key=lambda d: -cost[d])
    # Full-batch training needs about 40 * cost bytes: datasets above 1 GB run alone with
    # every thread, the others in at most MAX_WORKERS processes.
    big = [d for d in order if 40 * cost[d] > 1e9]
    for d in big:
        print(f"{d}: {40 * cost[d] / 1e9:.1f} GB, run alone", flush=True)
        run_and_save(d, os.cpu_count() or 4, run_fn, folder)
    n_jobs = min(n_jobs, MAX_WORKERS)
    n_threads = max(1, (os.cpu_count() or 4) // n_jobs)
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_and_save)(d, n_threads, run_fn, folder) for d in order if d not in big)
    pd.concat([pd.read_csv(f"{OUT}/{folder}/{d}.csv") for d in DATASETS]).to_csv(
        f"{OUT}/{name}.csv", index=False)
    np.savez(f"{OUT}/{name}_examples.npz",
             **{f"{d}/{k}": v for d in DATASETS for k, v in np.load(f"{OUT}/{folder}/{d}.npz").items()})


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_all(int(sys.argv[1]) if len(sys.argv) > 1 else 1, run, "forecast")
