# Soft-DTW mini-project (MVA — ML for Time Series)

Re-implementation and analysis of Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series* (ICML 2017), plus an extension: soft-DTW with a warping penalty ω on time-stretching steps (soft-DTW_ω).

Pablo Borrego, Pol Tibau.

## Contents
| File | Content |
|---|---|
| `notebook.ipynb` | **Main notebook**: method and verification, data diagnosis, all experiments (rebuilt from `results/`), extension |
| `report/main.tex`, `report/main.pdf` | Report |
| `softdtw.py` | Soft-DTW forward (Algorithm 1) and backward (Algorithm 2) in Numba, gradient, hard DTW and its path |
| `softdp.py` | Soft dynamic program with per-step costs: soft-DTW and soft-DTW_ω (`kind="adtw"`) |
| `torch_sdtw.py` | PyTorch loss `SoftDTWLoss(gamma, omega)` with a hand-written backward |
| `torch_sdtw_diag.py` | Pure-PyTorch anti-diagonal version (GPU-ready); `bench_gpu.py` benchmarks it (e.g. on Colab) |
| `barycenter.py` | Soft-DTW barycenters (L-BFGS), DBA (`tslearn`), batch subgradient method |
| `omega.py` | Barycenters and pairwise distances for soft-DTW_ω |
| `clustering.py` | k-means and nearest-centroid classifier |
| `tests.py`, `tests_softdp.py` | Test-suite (brute-force enumeration of alignments, finite differences, `tslearn`, autograd) |
| `data_diagnosis.py` | Data section: within-class lag and DTW gain |
| `make_figures.py` | Qualitative figures of the report |

## Experiments
Each script writes a CSV in `results/` and checkpoints every job in `results/*_parts/`, so an interrupted run resumes where it stopped. They take from minutes to hours on a laptop; the notebook only reads their results.

| Experiment (paper) | soft-DTW, DBA, subgradient | soft-DTW_ω |
|---|---|---|
| Barycenters (Table 1) | `exp_barycenters.py` | `exp_barycenters_omega.py` (`extra` for γ = 1, 0.001) |
| k-means (Table 2) | `exp_kmeans.py` | `exp_kmeans_omega.py` |
| Nearest centroid (Fig. 7) | `exp_classification.py` | `exp_classification_omega.py` |
| Forecasting (Table 3) | `exp_forecast.py` | `exp_forecast_omega.py` (γ = 0.01), `exp_forecast_omega_gammas.py` |

## Setup (Windows, Python ≥ 3.11)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy numba torch matplotlib pandas joblib tslearn scikit-learn pytest ipykernel nbformat nbclient
.venv/Scripts/python -m ipykernel install --user --name softdtw --display-name "Python (softdtw)"
.venv/Scripts/python -m pytest tests.py tests_softdp.py -q   # should print 67 passed
```
UCR datasets are downloaded automatically by `tslearn` on first use. Do not run two experiments at the same time (each one already uses all cores).

For the report: a LaTeX distribution (MiKTeX or TeX Live) and the VS Code extension *LaTeX Workshop*; `.vscode/settings.json` builds `report/main.tex` on save.
