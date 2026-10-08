"""Timing of forward + backward for a batch: Numba (CPU) vs anti-diagonal PyTorch (CPU / CUDA).

Run locally or on Google Colab with a GPU runtime (upload softdtw.py, torch_sdtw.py,
torch_sdtw_diag.py and this file, then `!python bench_gpu.py`).
Writes results/bench_gpu.csv.
"""

import os
import time

import pandas as pd
import torch

from torch_sdtw import SoftDTWLoss
from torch_sdtw_diag import SoftDTWLossDiag


def time_it(loss, x, y, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        xx = x.detach().clone().requires_grad_(True)
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss(xx, y).backward()
        if x.is_cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for B in (1, 32, 256):
        for n in (64, 128, 256, 512):
            x = torch.randn(B, n, 1, dtype=torch.float64)
            y = torch.randn(B, n, 1, dtype=torch.float64)
            rows.append(dict(B=B, n=n, impl="numba (cpu)", time=time_it(SoftDTWLoss(1.0), x, y)))
            for dev in devices:
                xd, yd = x.to(dev), y.to(dev)
                rows.append(dict(B=B, n=n, impl=f"anti-diagonal torch ({dev})",
                                 time=time_it(SoftDTWLossDiag(1.0), xd, yd)))
            print(rows[-1], flush=True)
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/bench_gpu.csv", index=False)
    print(df.pivot_table(index=["B", "n"], columns="impl", values="time").round(4))
