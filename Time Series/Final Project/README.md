# Soft-DTW project

Re-implementation and analysis of Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series* (ICML 2017), plus an extension: soft-DTW with a warping penalty ω on time-stretching steps (soft-DTW_ω).

Pablo Borrego, Pol Tibau.

## Contents
```
notebook.ipynb        main notebook: method and verification, data diagnosis,
                      all experiments (rebuilt from results/), extension
report/               report (main.tex, main.pdf)
softdtw/              the implementation
  core.py             forward (Algorithm 1) and backward (Algorithm 2) in Numba,
                      gradient, hard DTW and its path
  softdp.py           soft dynamic program with per-step costs: soft-DTW and
                      soft-DTW_ω (kind="adtw")
  torch_sdtw.py       PyTorch loss SoftDTWLoss(gamma, omega), hand-written backward
  torch_sdtw_diag.py  pure-PyTorch anti-diagonal version (GPU-ready)
  barycenter.py       soft-DTW barycenters (L-BFGS), DBA (tslearn), subgradient method
  omega.py            barycenters and pairwise distances for soft-DTW_ω
  clustering.py       k-means and nearest-centroid classifier
experiments/          one script per experiment (see below), data_diagnosis.py,
                      make_figures.py, bench_gpu.py
tests/                test-suite: brute-force enumeration of alignments, finite
                      differences, tslearn, autograd
results/              CSV results of the experiments
figures/              figures of the report
```

## Experiments
Run them from the project root, e.g. `python -m experiments.exp_kmeans`. Each script writes a CSV in `results/` and checkpoints every job in `results/*_parts/`, so an interrupted run resumes where it stopped. They take from minutes to hours on a laptop; the notebook only reads their results.

| Experiment (paper) | soft-DTW, DBA, subgradient | soft-DTW_ω |
|---|---|---|
| Barycenters (Table 1) | `exp_barycenters` | `exp_barycenters_omega` (argument `extra` for γ = 1, 0.001) |
| k-means (Table 2) | `exp_kmeans` | `exp_kmeans_omega` |
| Nearest centroid (Fig. 7) | `exp_classification` | `exp_classification_omega` |
| Forecasting (Table 3) | `exp_forecast` | `exp_forecast_omega` (γ = 0.01), `exp_forecast_omega_gammas` |

Also: `python -m experiments.data_diagnosis` (Data section) and `python -m experiments.make_figures` (qualitative figures).

## Setup (Windows, Python ≥ 3.11)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy numba torch matplotlib pandas joblib tslearn scikit-learn pytest ipykernel nbformat nbclient
.venv/Scripts/python -m ipykernel install --user --name softdtw --display-name "Python (softdtw)"
.venv/Scripts/python -m pytest tests -q   # should print 67 passed
```
UCR datasets are downloaded automatically by `tslearn` on first use. Do not run two experiments at the same time (each one already uses all cores).

For the report: a LaTeX distribution (MiKTeX or TeX Live) and the VS Code extension *LaTeX Workshop*; `.vscode/settings.json` builds `report/main.tex` on save.
