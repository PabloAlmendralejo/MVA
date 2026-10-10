# Soft-DTW project

Re-implementation and analysis of Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series* (ICML 2017), plus an extension: soft-DTW with a warping penalty ω on time-stretching steps (soft-DTW_ω).

## Setup (Windows, Python ≥ 3.11)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy numba torch matplotlib pandas joblib tslearn scikit-learn pytest ipykernel nbformat nbclient
.venv/Scripts/python -m ipykernel install --user --name softdtw --display-name "Python (softdtw)"
.venv/Scripts/python -m pytest tests.py tests_softdp.py -q   # should print 67 passed
```
UCR datasets are downloaded automatically by `tslearn` on first use.

For the report: a LaTeX distribution (MiKTeX or TeX Live) and the VS Code extension *LaTeX Workshop*; `.vscode/settings.json` builds `report/main.tex` on save.

## Working together
- Pull before you start (`git pull`), commit and push when you finish a piece of work.
- Avoid editing the same report section at the same time.
- Experiment results are committed in `results/`; long experiments resume from their checkpoints in `results/*_parts/`.
