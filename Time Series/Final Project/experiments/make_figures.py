"""Qualitative figures of the paper, reproduced (appendix of the report).

  figures/barycenters_ECG200.png  (paper Fig. 5): soft-DTW (gamma = 1, 0.01), soft-DTW_omega,
                                  DBA and subgradient barycenters of 10 ECG200 series, random init
  figures/interpolation_GunPoint.png (paper Fig. 4): weighted barycenters of two
                                  GunPoint series, Euclidean vs soft-DTW vs soft-DTW_omega (gamma = 1)
  figures/kmeans_CBF.png          (paper Fig. 6): k-means on CBF, soft-DTW and
                                  soft-DTW_omega (gamma = 1) vs DBA
  figures/prior_zero_cost.png     expected alignment E with zero costs: the
                                  data-independent prior of soft-DTW vs soft-DTW_omega
Usage:  python -m experiments.make_figures
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from softdtw.barycenter import dba_barycenter, softdtw_barycenter, subgradient_barycenter
from softdtw.clustering import dtw_kmeans
from softdtw.omega import soft_barycenter
from softdtw.softdp import edge_costs, soft_dp

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9})


def load(name):
    from tslearn.datasets import UCR_UEA_datasets
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return X.astype(np.float64), y


def barycenters_figure(keep=None, out="figures/barycenters_ECG200.png"):
    """keep: titles of the panels to draw (all five by default)."""
    X, y = load("ECG200")
    rng = np.random.default_rng(0)
    Y = list(X[rng.choice(np.where(y == np.unique(y)[0])[0], 10, replace=False)])
    init = rng.normal(size=Y[0].shape)
    methods = {"soft-DTW, $\\gamma=1$": lambda: softdtw_barycenter(Y, 1.0, init),
               "soft-DTW, $\\gamma=0.01$": lambda: softdtw_barycenter(Y, 0.01, init),
               "soft-DTW$_\\omega$, $\\gamma=\\omega=0.1$":
                   lambda: soft_barycenter(Y, "adtw", 0.1, init, omega=0.1).reshape(init.shape),
               "DBA": lambda: dba_barycenter(Y, init),
               "subgradient": lambda: subgradient_barycenter(Y, init)}
    if keep is not None:
        methods = {k: methods[k] for k in keep}
    fig, axes = plt.subplots(1, len(methods), figsize=(2.8 * len(methods), 2.4), sharey=True)
    for ax, (title, fn) in zip(axes, methods.items()):
        for s in Y:
            ax.plot(s[:, 0], color="0.75", lw=0.7)
        ax.plot(fn()[:, 0], color="C3", lw=1.8)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def barycenters_slides_figure():
    """Slides version: soft-DTW (gamma = 1, 0.01) and DBA only."""
    barycenters_figure(keep=["soft-DTW, $\\gamma=1$", "soft-DTW, $\\gamma=0.01$", "DBA"],
                       out="figures/barycenters_ECG200_slides.png")


def interpolation_figure():
    X, y = load("GunPoint")
    a, b = X[np.where(y == np.unique(y)[0])[0][0]], X[np.where(y == np.unique(y)[1])[0][0]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.6), sharey=True)
    for ax, title in zip(axes, ["Euclidean", "soft-DTW, $\\gamma=1$",
                                "soft-DTW$_\\omega$, $\\gamma=\\omega=1$"]):
        ax.plot(a[:, 0], color="C3", lw=1.5)
        ax.plot(b[:, 0], color="C0", lw=1.5)
        for w in (0.25, 0.5, 0.75):
            if title == "Euclidean":
                z = w * a + (1 - w) * b
            elif title.startswith("soft-DTW,"):
                z = softdtw_barycenter([a, b], 1.0, w * a + (1 - w) * b, weights=[w, 1 - w])
            else:
                z = soft_barycenter([a, b], "adtw", 1.0, w * a + (1 - w) * b, omega=1.0,
                                      weights=[w, 1 - w]).reshape(a.shape)
            ax.plot(z[:, 0], color="0.3", lw=1, ls="--")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("figures/interpolation_GunPoint.png", dpi=200)
    plt.close()


def kmeans_figure():
    X, y = load("CBF")
    k = len(np.unique(y))
    init = X[np.random.default_rng(0).choice(len(X), k, replace=False)]
    fig, axes = plt.subplots(3, k, figsize=(10, 6), sharey=True)
    for row, (title, method, gamma, omega) in enumerate([
            ("soft-DTW, $\\gamma=1$", "soft-DTW", 1.0, 0.0),
            ("soft-DTW$_\\omega$, $\\gamma=\\omega=1$", "soft-DTW_omega", 1.0, 1.0),
            ("DBA", "DBA", None, 0.0)]):
        C, labels = dtw_kmeans(X, k, method, init, gamma=gamma, omega=omega)
        for j in range(k):
            ax = axes[row, j]
            for s in X[labels == j]:
                ax.plot(s[:, 0], color="0.75", lw=0.6)
            ax.plot(C[j][:, 0], color="C3", lw=1.8)
            ax.set_title(f"{title}, cluster {j + 1}")
    plt.tight_layout()
    plt.savefig("figures/kmeans_CBF.png", dpi=200)
    plt.close()


def prior_figure():
    n = 60
    z = np.zeros((n, 1))  # identical constant series: all costs are zero
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.1))
    for ax, (title, kind, omega) in zip(axes, [("soft-DTW", "dtw", 0.0),
                                               ("soft-DTW$_\\omega$, $\\omega=0.5$", "adtw", 0.5),
                                               ("soft-DTW$_\\omega$, $\\omega=2$", "adtw", 2.0)]):
        Cd, Cv, Ch = edge_costs(z, z, kind, omega=omega)
        _, (Pd, Pv, Ph) = soft_dp(Cd, Cv, Ch, 1.0)
        ax.imshow((Pd + Pv + Ph)[1:, 1:], cmap="viridis")
        ax.set_title(title + ", $\\gamma=1$")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/prior_zero_cost.png", dpi=200)
    plt.close()


def nearest_centroid_slides_figure():
    """Accuracy of soft-DTW vs DBA centroids, one point per dataset (paper Fig. 7)."""
    import pandas as pd
    cl = pd.read_csv("results/classification.csv")
    acc = cl[cl.method.isin(["DBA", "soft-DTW"])].groupby(["dataset", "method"]).acc.mean().unstack()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot([0.4, 1], [0.4, 1], "k--", lw=0.8)
    ax.scatter(acc["DBA"], acc["soft-DTW"], color="C0")
    ax.set_xlabel("DBA centroids: accuracy")
    ax.set_ylabel("soft-DTW centroids: accuracy")
    plt.tight_layout()
    plt.savefig("figures/nearest_centroid_slides.png", dpi=200)
    plt.close()


FORECAST_CURVES = {"Euclidean": ("Euclidean loss", "C0"),
                   "soft-DTW": ("soft-DTW", "C3"),
                   "soft-DTW_omega 1": ("soft-DTW$_\\omega$ ($\\omega=1$)", "C2")}


def forecast_figure(examples=(("ShapesAll", 0), ("Coffee", 3), ("CBF", 1)),
                    curves=("soft-DTW", "soft-DTW_omega 1"),
                    out="figures/forecast_omega_examples.png"):
    z = np.load("results/forecast_omega_examples.npz")
    fig, axes = plt.subplots(1, len(examples), figsize=(10, 2.4))
    for ax, (name, k) in zip(axes, examples):
        inp, truth = z[f"{name}/input"][k], z[f"{name}/truth"][k]
        t = np.arange(len(inp) + len(truth))
        ax.plot(t[:len(inp)], inp, color="k", lw=1, label="input")
        ax.plot(t[len(inp):], truth, color="k", lw=1, ls="--", label="ground truth")
        for key in curves:
            label, color = FORECAST_CURVES[key]
            ax.plot(t[len(inp):], z[f"{name}/{key}"][k], color=color, lw=1.2, label=label)
        ax.set_title(name)
        ax.set_xticks([])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2 + len(curves),
               fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(out, dpi=200)
    plt.close()


def forecast_slides_figure():
    """Slides version, before soft-DTW_omega is introduced: Euclidean loss vs soft-DTW."""
    forecast_figure(curves=("Euclidean", "soft-DTW"), out="figures/forecast_examples_slides.png")


if __name__ == "__main__":
    for f in (barycenters_figure, barycenters_slides_figure, interpolation_figure, kmeans_figure, prior_figure, forecast_figure, forecast_slides_figure, nearest_centroid_slides_figure):
        f()
        print(f.__name__, "done", flush=True)
