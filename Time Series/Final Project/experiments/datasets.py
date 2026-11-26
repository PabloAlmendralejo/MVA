"""UCR datasets of each experiment of the paper (appendix tables G-L), and full-paper mode.

  Barycenters (Table 1):      79 datasets.
  k-means (Table 2):          57 (random init) / 58 (Euclidean init) datasets.
  Nearest centroid (Fig. 7):  no per-dataset table in the paper; we use the barycenter list.
  Forecasting (Table 3):      76 datasets.

Names are the 2018 / tslearn ones. SOFTDTW_FULL=1 switches every experiment to these lists
and writes to results_full/; results/ keeps the reduced runs of the report.
"""

import os

import pandas as pd

BARYCENTERS = [
    "FiftyWords", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY",
    "CricketZ", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "ECG200", "ECG5000", "ECGFiveDays",
    "Earthquakes", "ElectricDevices", "Fish", "FaceAll", "FaceFour", "FacesUCR", "FordA",
    "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate",
    "ItalyPowerDemand", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain",
    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OSULeaf", "OliveOil",
    "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices",
    "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry",
    "SwedishLeaf", "Symbols", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoPatterns",
    "TwoLeadECG", "UWaveGestureLibraryAll", "Wine", "WordSynonyms", "Worms", "WormsTwoClass",
    "SyntheticControl", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
    "Wafer", "Yoga",
]

_KMEANS_EXCLUDED = {
    "HandOutlines", "Haptics", "Herring", "InlineSkate", "ItalyPowerDemand",
    "NonInvasiveFetalECGThorax2", "OSULeaf", "OliveOil", "PhalangesOutlinesCorrect", "Phoneme",
    "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "Strawberry",
    "SwedishLeaf", "Symbols", "ToeSegmentation1", "ToeSegmentation2", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "Wafer", "Yoga",
}
KMEANS_RANDOM = [d for d in BARYCENTERS if d not in _KMEANS_EXCLUDED]
KMEANS_EUCLIDEAN = KMEANS_RANDOM + ["HandOutlines"]

CLASSIFICATION = BARYCENTERS

_FORECAST_EXCLUDED = {"FordA", "FordB", "HandOutlines", "InlineSkate",
                      "NonInvasiveFetalECGThorax2", "ScreenType", "SmallKitchenAppliances",
                      "StarLightCurves"}
FORECAST = [d for d in BARYCENTERS if d not in _FORECAST_EXCLUDED] + [
    "InsectWingbeatSound", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Meat"]

assert (len(BARYCENTERS), len(KMEANS_RANDOM), len(KMEANS_EUCLIDEAN), len(FORECAST)) == (79, 57, 58, 76)

FULL = os.environ.get("SOFTDTW_FULL") == "1"
OUT = os.environ.get("SOFTDTW_OUT") or ("results_full" if FULL else "results")


def checkpointed(part, fn):
    """Rows of a job: read from the csv `part` if it exists, else computed by fn() and saved.

    The csv is written to a temporary file and renamed, so an interrupted job leaves no
    truncated checkpoint.
    """
    if os.path.exists(part):
        return pd.read_csv(part).to_dict("records")
    rows = fn()
    os.makedirs(os.path.dirname(part), exist_ok=True)
    pd.DataFrame(rows).to_csv(part + ".tmp", index=False)
    os.replace(part + ".tmp", part)
    return rows
