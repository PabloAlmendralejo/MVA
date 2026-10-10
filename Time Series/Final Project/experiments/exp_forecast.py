"""Table 3 / Figure 1 of the paper (reduced): multistep-ahead prediction with an MLP.

Protocol (Section 4.4): UCR train/test splits; the first 60% of each series is the
input, the remaining 40% the output (class labels ignored). One-hidden-layer MLP
with sigmoid activation, trained with Adam, under the Euclidean loss or the
soft-DTW loss (gamma in {1, 0.1, 0.01, 0.001}). Two initialisations of the
soft-DTW models: random, or the weights of the Euclidean-trained MLP.
Test metrics: Euclidean (sum of squares per series) and hard DTW; for each
dataset and metric the 5 models are ranked, then ranks are averaged.

Usage:  python -m experiments.exp_forecast  ->  results/forecast.csv, results/forecast_examples.npz
"""

import copy
import warnings

import numpy as np
import pandas as pd
import torch

from softdtw.clustering import pairwise_sdtw
from softdtw.torch_sdtw import SoftDTWLoss

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "CBF", "ECGFiveDays", "GunPoint", "Plane", "ArrowHead", "Coffee",
            "Trace", "MedicalImages", "ShapesAll"]
GAMMAS = [1.0, 0.1, 0.01, 0.001]
HIDDEN, EPOCHS, LR, SEED = 256, 400, 1e-3, 0
torch.set_num_threads(4)


def load(name):
    from tslearn.datasets import UCR_UEA_datasets
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
            rows.append(dict(dataset=name, init=init, model=f"soft-DTW", gamma=g, euclid=e, dtw=d))
            if init == "random":
                preds[f"soft-DTW {g}"] = P
    return rows, preds, Tin.numpy(), Tout.numpy()


if __name__ == "__main__":
    import os
    warnings.filterwarnings("ignore")
    os.makedirs("results/forecast_parts", exist_ok=True)
    for name in DATASETS:  # each dataset is checkpointed; finished ones are skipped
        part = f"results/forecast_parts/{name}"
        if os.path.exists(part + ".csv"):
            continue
        rows, preds, Tin, Tout = run(name)
        k = slice(0, 5)  # keep a few test series for the figures
        ex = {"input": Tin[k], "truth": Tout[k], **{m: P[k] for m, P in preds.items()}}
        np.savez(part + ".npz", **ex)
        pd.DataFrame(rows).to_csv(part + ".csv", index=False)
        print(name, "done", flush=True)
    pd.concat([pd.read_csv(f"results/forecast_parts/{d}.csv") for d in DATASETS]).to_csv(
        "results/forecast.csv", index=False)
    examples = {f"{d}/{key}": v for d in DATASETS
                for key, v in np.load(f"results/forecast_parts/{d}.npz").items()}
    np.savez("results/forecast_examples.npz", **examples)
