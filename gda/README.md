# Convolutional Wasserstein Distances — GDA project (MVA 2026)

Reproduction of **Solomon, de Goes, Peyré, Cuturi, Butscher, Nguyen, Du, Guibas,
*Convolutional Wasserstein Distances: Efficient Optimal Transportation on Geometric
Domains*, SIGGRAPH 2015**, for the Geometric Data Analysis course (J. Feydy).

Idea of the paper: in entropic OT the only thing you ever do with the cost is apply the
kernel `exp(-d(x,y)^2 / gamma)`. By Varadhan's formula that kernel is approximately the
heat kernel at time `gamma/2`, and the heat kernel can be applied in near-linear time
(Gaussian convolution on a grid, one prefactored sparse solve on a mesh). So Sinkhorn
never has to build an `n x n` matrix.

## Setup

```bash
pip install -r requirements.txt
```

Run everything from the repo root so that `import gda` works:

```bash
python experiments/A_sinkhorn/your_script.py
```

## Layout

```
gda/                  shared code -- empty stubs for now, keep it small
  sinkhorn.py         sinkhorn() + barycenter()                      (A)
  kernels.py          gaussian_grid_kernel() (B), implicit_heat_kernel() (C)
experiments/
  A_sinkhorn/         Sinkhorn + entropic regularisation
  B_heat_kernel/      heat kernel on grids
  C_meshes/           meshes & geometry
figures/              generated figures (png/pdf), referenced from the report
data/meshes/          input meshes (not committed)
report/               LaTeX source for the report (Overleaf)
```

## Who does what

| | topic | lives in |
|---|---|---|
| A | Sinkhorn, entropic regularisation, effect of `gamma`, comparison against exact OT, entropic sharpening | `experiments/A_sinkhorn/` |
| B | heat kernel on regular grids: Gaussian convolution vs. implicit Euler, substeps, cost of the approximation | `experiments/B_heat_kernel/` |
| C | meshes: cotangent Laplacian, mesh quality, heat leaking between nearby parts, translated-shapes test | `experiments/C_meshes/` |

One folder per person. Each of us also writes the corresponding section
of the report.

The files in `gda/` are the only shared code. A fills in
`sinkhorn.py`, B and C their own kernel. Anything that is only useful for your own
experiments goes in your own folder, not in `gda/`.
