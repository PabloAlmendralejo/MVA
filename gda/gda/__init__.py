"""Convolutional Wasserstein distances — GDA project (MVA 2026)."""
from .sinkhorn import sinkhorn, barycenter
from .kernels import gaussian_grid_kernel, implicit_heat_kernel
