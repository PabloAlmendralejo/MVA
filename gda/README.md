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

## The one interface we share

Everything goes through a kernel-application function. A kernel module provides

```python
apply_K, a = some_kernel(...)   # a = area weights, a @ mu == 1 for a density mu
# apply_K(v)[i] == sum_j K[i,j] * a[j] * v[j]
```

and then the solver is kernel-agnostic:

```python
dist2, v, w = sinkhorn(mu0, mu1, apply_K, a, gamma)
mu = barycenter([mu0, mu1], [0.5, 0.5], apply_K, a)
```

Please don't change that signature without telling the others — it is the only thing
that couples the three parts.

## Who does what

| | topic | lives in |
|---|---|---|
| A | Sinkhorn, entropic regularisation, effect of `gamma`, comparison against exact OT, entropic sharpening | `experiments/A_sinkhorn/` |
| B | heat kernel on regular grids: Gaussian convolution vs. implicit Euler, substeps, cost of the approximation | `experiments/B_heat_kernel/` |
| C | meshes: cotangent Laplacian, mesh quality, heat leaking between nearby parts, translated-shapes test | `experiments/C_meshes/` |

One folder per person, but named after the topic rather than after us, so that swapping
parts does not mean renaming anything. Each of us also writes the corresponding section
of the report.

The stubs in `gda/` are the only shared code: the signatures are fixed, A fills in
`sinkhorn.py`, B and C their own kernel. Anything that is only useful for your own
experiments goes in your own folder, not in `gda/`.

## Conventions

- Toy examples we fully control, not benchmarks. Every figure must be reproducible by
  running one script in `experiments/`.
- Figures: save to `figures/` as `png` at `dpi=150` (or `pdf`), named after the script
  that produced them. Default matplotlib style, no titles inside the figure (the caption
  does that job).
- Densities are always normalised w.r.t. the area weights: `a @ mu == 1`.
- Commit messages: `A: ...`, `B: ...`, `C: ...` so it is obvious who touched what.

## Two traps, so nobody loses an afternoon to them

- `scipy.ndimage.gaussian_filter` truncates at `4*sigma` by default, i.e. it sets the
  kernel to *exactly* zero further away, and then Sinkhorn divides by zero. Pass a large
  `truncate`.
- Plain (non-log-domain) Sinkhorn overflows `float64` once `gamma` is small relative to
  the transport distance, because the scalings grow like `exp(d^2 / gamma)`. Log-domain
  stabilisation is incompatible with the convolution trick, so this is a real limitation
  of the method rather than a bug — worth a paragraph in the report.
