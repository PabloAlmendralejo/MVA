"""Data diagnosis for the Data section: size, normalisation, and how much
temporal misalignment there is within each class (the property DTW exploits).

For random pairs of series of the same class we measure
  * lag: the shift maximising their cross-correlation, in % of the length;
  * gain: 1 - DTW / Euclidean (squared costs), the share of the Euclidean cost
    removed by allowing time warping;
  * stretch: share of the steps of the optimal DTW path that stretch time (not
    diagonal), i.e. the steps that the penalty omega charges.
Usage:  python -m experiments.data_diagnosis  ->  results/data_diagnosis.csv
        python -m experiments.data_diagnosis full  ->  results/data_diagnosis_full.csv
The first run uses 13 datasets, the second the 84 datasets of the paper.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from tslearn.datasets import UCR_UEA_datasets

from softdtw import dtw_path

DATASETS = ["ItalyPowerDemand", "SyntheticControl", "TwoLeadECG", "ECG200",
            "MedicalImages", "CBF", "ECGFiveDays", "Plane", "GunPoint",
            "ArrowHead", "Trace", "Coffee", "ShapesAll"]
# the other datasets of the paper (barycenter and forecasting lists)
OTHERS = """
Adiac Beef BeetleFly BirdChicken Car ChlorineConcentration CinCECGTorso Computers
CricketX CricketY CricketZ DiatomSizeReduction DistalPhalanxOutlineAgeGroup
DistalPhalanxOutlineCorrect DistalPhalanxTW ECG5000 Earthquakes ElectricDevices
FaceAll FaceFour FacesUCR FiftyWords Fish FordA FordB Ham HandOutlines Haptics
Herring InlineSkate InsectWingbeatSound LargeKitchenAppliances Lightning2 Lightning7
Meat MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW
MoteStrain NonInvasiveFetalECGThorax1 NonInvasiveFetalECGThorax2 OSULeaf OliveOil
PhalangesOutlinesCorrect Phoneme ProximalPhalanxOutlineAgeGroup
ProximalPhalanxOutlineCorrect ProximalPhalanxTW RefrigerationDevices ScreenType
ShapeletSim SmallKitchenAppliances SonyAIBORobotSurface1 SonyAIBORobotSurface2
StarLightCurves Strawberry SwedishLeaf Symbols ToeSegmentation1 ToeSegmentation2
TwoPatterns UWaveGestureLibraryAll UWaveGestureLibraryX UWaveGestureLibraryY
UWaveGestureLibraryZ Wafer Wine WordSynonyms Worms WormsTwoClass Yoga
""".split()
N_PAIRS = 200


def load_all(name):
    X, y, Xt, yt = UCR_UEA_datasets(use_cache=True).load_dataset(name)
    return (np.concatenate([X, Xt])[..., 0].astype(np.float64),
            np.concatenate([y, yt]), len(X), len(Xt))


def best_lag(a, b):
    a, b = a - a.mean(), b - b.mean()
    c = np.correlate(a, b, mode="full")
    return abs(int(np.argmax(c)) - (len(b) - 1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    full = "full" in sys.argv[1:]
    out = "results/data_diagnosis_full.csv" if full else "results/data_diagnosis.csv"
    rows = []
    for name in DATASETS + (OTHERS if full else []):
        X, y, ntr, nte = load_all(name)
        n = X.shape[1]
        lags, gains, stretches = [], [], []
        classes = np.unique(y)
        for _ in range(N_PAIRS):
            c = rng.choice(classes)
            members = np.where(y == c)[0]
            if len(members) < 2:
                continue
            i, j = rng.choice(members, 2, replace=False)
            lags.append(100 * best_lag(X[i], X[j]) / n)
            eu = ((X[i] - X[j]) ** 2).sum()
            if eu == 0:  # identical series: no cost to remove
                continue
            A, d = dtw_path(X[i], X[j])
            gains.append(1 - d / eu)
            # path with L cells, L - 1 steps: 2 (L - n) of them are not diagonal
            L = A.sum()
            stretches.append(100 * 2 * (L - n) / (L - 1))
        rows.append(dict(dataset=name, n_train=ntr, n_test=nte, length=n,
                         n_classes=len(classes),
                         mean_abs_mean=float(np.abs(X.mean(1)).mean()),
                         mean_std=float(X.std(1).mean()),
                         lag_median_pct=float(np.median(lags)),
                         lag_p90_pct=float(np.percentile(lags, 90)),
                         dtw_gain_median=float(np.median(gains)),
                         stretch_median_pct=float(np.median(stretches))))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
