"""Summary tables of the full-paper runs (results_full/), next to the paper's values.

Usage:  python -m experiments.summarize_full  ->  results_full/summary/*.csv

  table1_barycenters_vs_paper.csv   % of datasets where soft-DTW has a lower mean hard-DTW
                                    loss than DBA / subgradient (paper Table 1, rounded)
  table3_forecast_vs_paper.csv      average rank of each training loss (paper Table 3)
  fig7_classification_by_dataset.csv  mean test accuracy over the 3 splits, selected gammas
  barycenters_omega_summary.csv     soft-DTW_omega vs DBA and vs soft-DTW (train / held-out)
  forecast_omega_ranks.csv          Table 3 ranks when soft-DTW_omega replaces soft-DTW (same gamma)
  forecast_omega_vs_softdtw.csv     soft-DTW_omega vs soft-DTW (same gamma): errors and timing
  forecast_omega_vs_euclidean_net.csv  soft-DTW and soft-DTW_omega networks vs the Euclidean network,
                                    on both metrics at once
  classification_omega_by_dataset.csv  mean test accuracy: soft-DTW (15 / 5 gammas), soft-DTW_omega,
                                    DBA, Euclidean NC
  classification_omega_summary.csv  each method vs soft-DTW on the same 5 gammas
  table2_kmeans_subset_vs_paper.csv % of datasets where soft-DTW has a lower k-means energy than
                                    DBA / subgradient (paper Table 2; our 41-dataset subset)
  kmeans_subset_omega_summary.csv   soft-DTW_omega vs soft-DTW and DBA: energy and ARI / NMI
"""

import os

os.environ.setdefault("SOFTDTW_FULL", "1")  # summarises results_full/; before experiments.* is imported

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.datasets import OUT

G = [1.0, 0.1, 0.01, 0.001]
PAPER1 = {("DBA", "random"): [41, 94, 100, 97], ("DBA", "euclidean"): [4, 47, 80, 90],
          ("subgradient", "random"): [96, 97, 97, 97], ("subgradient", "euclidean"): [35, 72, 92, 97]}
PAPER3 = {("dtw", "random"): [3.46, 3.55, 3.33, 2.79, 1.87],
          ("dtw", "euclidean"): [4.21, 3.96, 3.42, 2.12, 1.29],
          ("euclid", "random"): [1.05, 2.41, 3.42, 4.13, 3.99],
          ("euclid", "euclidean"): [1.70, 2.99, 3.38, 3.64, 3.29]}
PAPER2 = {("DBA", "random"): [16, 25, 60, 77], ("DBA", "euclidean"): [29, 24, 55, 69],
          ("subgradient", "random"): [42, 58, 76, 96], ("subgradient", "euclidean"): [46, 50, 66, 84]}
OMEGAS = [0.1, 1.0]


def forecast_omega_ranks(src):
    """Average rank among {Euclidean net, 4 gammas}, with soft-DTW (omega = 0) or soft-DTW_omega."""
    f = pd.read_csv(src)
    f["omega"] = f.omega.fillna(0.0)
    rows = []
    for metric in ["dtw", "euclid"]:
        for init in ["random", "euclidean"]:
            x = f[f.init == init]
            for om in [0.0] + OMEGAS:
                cols = {"Euclidean": x[x.model == "Euclidean"].set_index("dataset")[metric]}
                for g in G:
                    cols[g] = x[(x.model != "Euclidean") & (x.gamma == g) & (x.omega == om)].set_index("dataset")[metric]
                rank = pd.DataFrame(cols).rank(axis=1).mean()
                rows.append(dict(metric=metric, init=init, omega=om, euclidean_net=round(rank["Euclidean"], 2),
                                 **{f"gamma={g}": round(rank[g], 2) for g in G}))
    return pd.DataFrame(rows)


def forecast_omega_vs_softdtw(src):
    f = pd.read_csv(src)
    f["omega"] = f.omega.fillna(0.0)
    rows = []
    for init in ["random", "euclidean"]:
        x = f[f.init == init]
        p = x[x.model != "Euclidean"].pivot_table(index="dataset", columns=["gamma", "omega"],
                                                  values=["dtw", "euclid", "t_max_err", "t_min_err"])
        eu = x[x.model == "Euclidean"].set_index("dataset").reindex(p.index)
        for g in G:
            for om in [0.0] + OMEGAS:
                row = dict(init=init, gamma=g, omega=om,
                           median_t_max_err_pct=round(float(p[("t_max_err", g, om)].median()), 2),
                           median_t_min_err_pct=round(float(p[("t_min_err", g, om)].median()), 2),
                           n_datasets_dtw_over_1p5x_euclidean_net=int((p[("dtw", g, om)] > 1.5 * eu.dtw).sum()),
                           n_datasets_euclid_over_10x_euclidean_net=int((p[("euclid", g, om)] > 10 * eu.euclid).sum()))
                if om > 0:
                    row.update(median_dtw_ratio_vs_softDTW=round(float((p[("dtw", g, om)] / p[("dtw", g, 0.0)]).median()), 3),
                               median_euclid_ratio_vs_softDTW=round(float((p[("euclid", g, om)] / p[("euclid", g, 0.0)]).median()), 3),
                               pct_datasets_lower_dtw=round(100 * float((p[("dtw", g, om)] < p[("dtw", g, 0.0)]).mean()), 1),
                               pct_datasets_lower_euclid=round(100 * float((p[("euclid", g, om)] < p[("euclid", g, 0.0)]).mean()), 1))
                rows.append(row)
    return pd.DataFrame(rows)


def forecast_vs_euclidean_net(src):
    """Each soft-DTW / soft-DTW_omega network against the Euclidean network, on both metrics."""
    f = pd.read_csv(src)
    f["omega"] = f.omega.fillna(0.0)
    rows = []
    for init in ["random", "euclidean"]:
        x = f[f.init == init]
        eu = x[x.model == "Euclidean"].set_index("dataset")
        for g in G:
            for om in [0.0] + OMEGAS:
                s = x[(x.model != "Euclidean") & (x.gamma == g) & (x.omega == om)].set_index("dataset").reindex(eu.index)
                rows.append(dict(init=init, gamma=g, omega=om,
                                 median_dtw_ratio_vs_euclidean_net=round(float((s.dtw / eu.dtw).median()), 3),
                                 pct_datasets_lower_dtw=round(100 * float((s.dtw < eu.dtw).mean()), 1),
                                 median_euclid_ratio_vs_euclidean_net=round(float((s.euclid / eu.euclid).median()), 3),
                                 pct_datasets_lower_euclid=round(100 * float((s.euclid < eu.euclid).mean()), 1),
                                 pct_datasets_lower_on_both=round(100 * float(((s.dtw < eu.dtw) & (s.euclid < eu.euclid)).mean()), 1)))
    return pd.DataFrame(rows)


def _select(df, keys):
    """Best validation accuracy per (dataset, split); smallest gamma (then omega) on ties."""
    df = df.sort_values(list(keys))
    return df.loc[df.groupby(["dataset", "split"]).val_acc.idxmax()]


def classification_omega_by_dataset(src):
    a = pd.read_csv(os.path.join(os.path.dirname(src), "classification_all_gammas.csv"))
    co = pd.read_csv(src)
    sd = a[a.method == "soft-DTW"]
    sd5 = sd[np.isclose(sd.gamma.values[:, None], sorted(co.gamma.unique())).any(1)]
    base = a[a.method != "soft-DTW"].groupby(["dataset", "method"]).acc.mean().unstack()
    acc = lambda df, keys: _select(df, keys).groupby("dataset").acc.mean()
    return pd.DataFrame({"soft-DTW_15_gammas": acc(sd, ["gamma"]), "soft-DTW_5_gammas": acc(sd5, ["gamma"]),
                         "soft-DTW_omega_5_gammas_x_2_omegas": acc(co, ["gamma", "omega"]),
                         "soft-DTW_omega_0.1_5_gammas": acc(co[co.omega == 0.1], ["gamma"]),
                         "soft-DTW_omega_1_5_gammas": acc(co[co.omega == 1.0], ["gamma"]),
                         "DBA": base["DBA"], "Euclidean_NC": base["Euclidean NC"]}).round(4)


def classification_omega_summary(src):
    t = classification_omega_by_dataset(src)
    ref = t["soft-DTW_5_gammas"]
    rows = []
    for c in t.columns:
        row = dict(method=c, mean_acc=round(float(t[c].mean()), 4))
        if c != "soft-DTW_5_gammas":
            row.update(pct_datasets_better=round(100 * float((t[c] > ref + 1e-12).mean()), 1),
                       pct_datasets_tied=round(100 * float((abs(t[c] - ref) <= 1e-12).mean()), 1),
                       pct_datasets_worse=round(100 * float((t[c] < ref - 1e-12).mean()), 1),
                       mean_diff=round(float((t[c] - ref).mean()), 4),
                       wilcoxon_p=round(float(wilcoxon(t[c], ref, zero_method="zsplit").pvalue), 4))
        rows.append(row)
    return pd.DataFrame(rows)


def table2_subset(src):
    k = pd.read_csv(src)
    e = k.set_index(["dataset", "init", "method", "gamma"]).energy
    rows = []
    for (comp, init), paper in PAPER2.items():
        c = e.xs((init, comp), level=["init", "method"]).droplevel("gamma")
        for g, pv in zip(G, paper):
            s = e.xs((init, "soft-DTW", g), level=["init", "method", "gamma"])
            rows.append(dict(compared_with=comp, init=init, gamma=g,
                             ours_pct_41_datasets=round(100 * float((s < c.reindex(s.index)).mean()), 1),
                             paper_pct_57_58_datasets=pv))
    return pd.DataFrame(rows)


def kmeans_omega_summary(src):
    k = pd.read_csv(src)
    k["gamma"], k["omega"] = k.gamma.fillna(-1), k.omega.fillna(0.0)
    p = k.pivot_table(index=["dataset", "init"], columns=["method", "gamma", "omega"], values=["energy", "ari", "nmi"])
    rows = []
    for init in ["random", "euclidean"]:
        x = p.xs(init, level="init")
        for g in G:
            for om in OMEGAS:
                o, s = ("soft-DTW_omega", g, om), ("soft-DTW", g, 0.0)
                rows.append(dict(
                    init=init, gamma=g, omega=om,
                    mean_ari_omega=round(float(x[("ari",) + o].mean()), 3), mean_ari_softDTW=round(float(x[("ari",) + s].mean()), 3),
                    mean_ari_DBA=round(float(x[("ari", "DBA", -1, 0.0)].mean()), 3),
                    mean_nmi_omega=round(float(x[("nmi",) + o].mean()), 3), mean_nmi_softDTW=round(float(x[("nmi",) + s].mean()), 3),
                    ari_wilcoxon_p_vs_softDTW=round(float(wilcoxon(x[("ari",) + o], x[("ari",) + s], zero_method="zsplit").pvalue), 4),
                    median_energy_ratio_vs_softDTW=round(float((x[("energy",) + o] / x[("energy",) + s]).median()), 3),
                    pct_datasets_lower_energy_than_softDTW=round(100 * float((x[("energy",) + o] < x[("energy",) + s]).mean()), 1),
                    pct_datasets_lower_energy_than_DBA=round(100 * float((x[("energy",) + o] < x[("energy", "DBA", -1, 0.0)]).mean()), 1)))
    return pd.DataFrame(rows)


def table1(src):
    b = pd.read_csv(src)
    m = b.groupby(["dataset", "init", "method", "gamma"], dropna=False).loss.mean()
    rows = []
    for (comp, init), paper in PAPER1.items():
        c = m.xs((init, comp), level=["init", "method"]).droplevel("gamma")
        for g, pv in zip(G, paper):
            s = m.xs((init, "soft-DTW", g), level=["init", "method", "gamma"])
            rows.append(dict(compared_with=comp, init=init, gamma=g,
                             ours_pct=round(100 * (s < c.reindex(s.index)).mean(), 1), paper_pct=pv))
    return pd.DataFrame(rows)


def table3(src):
    f = pd.read_csv(src)
    rows = []
    for (metric, init), paper in PAPER3.items():
        x = f[f.init == init].copy()
        x["key"] = x.model.where(x.model == "Euclidean", "soft-DTW gamma=" + x.gamma.astype(str))
        p = x.pivot_table(index="dataset", columns="key", values=metric)
        rank = p.rank(axis=1).mean()
        for k, pv in zip(["Euclidean"] + [f"soft-DTW gamma={g}" for g in G], paper):
            rows.append(dict(metric=metric, init=init, model=k, ours_rank=round(rank[k], 2), paper_rank=pv,
                             median_error_ratio_vs_euclidean_net=round(float((p[k] / p["Euclidean"]).median()), 3)))
    return pd.DataFrame(rows)


def fig7(src):
    c = pd.read_csv(src)
    a = c.groupby(["dataset", "method"]).acc.mean().unstack()
    a["selected_gammas_soft-DTW"] = (c[c.method == "soft-DTW"].groupby("dataset").gamma
                                     .apply(lambda s: " / ".join(f"{v:.3g}" for v in s)))
    a["soft-DTW_minus_DBA"] = a["soft-DTW"] - a["DBA"]
    return a.round(4)


def omega_barycenters(src):
    w = pd.read_csv(src)
    w["gamma"], w["omega"] = w.gamma.fillna(-1), w.omega.fillna(0)  # baselines: no gamma; soft-DTW: omega = 0
    rows = []
    for loss in ["train_loss", "heldout_loss"]:
        mean = w.groupby(["dataset", "init", "method", "gamma", "omega"])[loss].mean()
        pairs = w.pivot_table(index=["dataset", "rep", "cls", "init"], columns=["method", "gamma", "omega"], values=loss)
        for init in ["random", "euclidean"]:
            dba = mean.xs((init, "DBA"), level=["init", "method"]).droplevel(["gamma", "omega"])
            x = pairs.xs(init, level="init")
            for om in [0.0, 0.1, 1.0]:
                meth = "soft-DTW" if om == 0 else "soft-DTW_omega"
                for g in G:
                    lv = ["init", "method", "gamma", "omega"]
                    s = mean.xs((init, meth, g, om), level=lv)
                    d = dba.reindex(s.index)
                    z = mean.xs((init, "soft-DTW", g, 0.0), level=lv)
                    ok = s.notna() & d.notna()  # held-out loss is NaN when a class has no spare series
                    rows.append(dict(
                        loss=loss, init=init, method=meth, gamma=g, omega=om,
                        pct_datasets_better_than_DBA=round(100 * (s[ok] < d[ok]).mean(), 1),
                        pct_datasets_better_than_softDTW_same_gamma=np.nan if om == 0 else round(100 * (s[ok] < z[ok]).mean(), 1),
                        median_ratio_vs_DBA=round(float((x[(meth, g, om)] / x[("DBA", -1, 0.0)]).median()), 3),
                        median_ratio_vs_softDTW=np.nan if om == 0 else round(float((x[(meth, g, om)] / x[("soft-DTW", g, 0.0)]).median()), 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = os.path.join(OUT, "summary")
    os.makedirs(out, exist_ok=True)
    jobs = [("barycenters.csv", table1, "table1_barycenters_vs_paper.csv", False),
            ("forecast.csv", table3, "table3_forecast_vs_paper.csv", False),
            ("classification.csv", fig7, "fig7_classification_by_dataset.csv", True),
            ("barycenters_omega.csv", omega_barycenters, "barycenters_omega_summary.csv", False),
            ("forecast_omega.csv", forecast_omega_ranks, "forecast_omega_ranks.csv", False),
            ("forecast_omega.csv", forecast_omega_vs_softdtw, "forecast_omega_vs_softdtw.csv", False),
            ("forecast_omega.csv", forecast_vs_euclidean_net, "forecast_omega_vs_euclidean_net.csv", False),
            ("classification_omega_all.csv", classification_omega_by_dataset, "classification_omega_by_dataset.csv", True),
            ("classification_omega_all.csv", classification_omega_summary, "classification_omega_summary.csv", False),
            ("kmeans_subset.csv", table2_subset, "table2_kmeans_subset_vs_paper.csv", False),
            ("kmeans_subset.csv", kmeans_omega_summary, "kmeans_subset_omega_summary.csv", False)]
    for src, fn, dst, index in jobs:
        if os.path.exists(os.path.join(OUT, src)):
            fn(os.path.join(OUT, src)).to_csv(os.path.join(out, dst), index=index)
            print("wrote", os.path.join(out, dst))
