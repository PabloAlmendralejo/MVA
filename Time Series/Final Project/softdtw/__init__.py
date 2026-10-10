"""Soft-DTW (Cuturi & Blondel, 2017) and soft-DTW with a warping penalty omega.

core: forward/backward recursions, gradient, hard DTW; softdp: soft DP with per-step costs;
torch_sdtw(_diag): PyTorch losses; barycenter, omega, clustering: barycenters, k-means,
nearest centroid.
"""

from .core import *  # noqa: F401,F403
