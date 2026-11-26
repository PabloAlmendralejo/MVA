# Soft-DTW: full reproduction of the paper (results)

This folder holds the results of reproducing Cuturi & Blondel (2017) on **the same UCR dataset lists as the paper**:
- barycenters: 79 datasets;
- forecasting: 76 datasets;
- classification: 79 datasets;
- k-means: 41 of the paper's 57 datasets (the CPU-affordable ones).

The report (`report/`) is based on a reduced run on 10–13 datasets, whose results are in `results/`.

It also holds the same experiments for our variant, **soft-DTW_omega** (soft-DTW with a penalty ω on every warping step), run on the same data, draws and initialisations as soft-DTW so that the two are directly comparable.

Protocols and code are those of the report, run with `SOFTDTW_FULL=1` (see the project README). The summary tables are produced by `python -m experiments.summarize_full`.

**Status:**

| Experiment | Status |
|---|---|
| Barycenters (Table 1) | Done |
| Forecasting (Table 3) | Done |
| Nearest-centroid classification (Fig. 7) | Done |
| k-means (Table 2) | Done on a subset: 41 of the paper's 57 datasets (Section 2.4) |
| Barycenters, soft-DTW_omega | Done |
| Forecasting, soft-DTW_omega | Done |
| Classification, soft-DTW_omega | Done |
| k-means, soft-DTW_omega | Done (same 41 datasets) |

---

## 1. Files

### Raw results: one row per run

**`barycenters.csv`**: Table 1, 9,480 rows.
- 79 datasets × 10 repetitions × 2 initialisations.
- Methods:
  - soft-DTW with γ ∈ {1, 0.1, 0.01, 0.001};
  - DBA (tslearn);
  - subgradient.
- Columns:
  - `dataset`, `length`;
  - `rep`: 0–9;
  - `cls`: the drawn class;
  - `init`: `random` = N(0,1), `euclidean` = Euclidean mean;
  - `method`, `gamma`;
  - `loss`: hard-DTW loss of the barycenter on the 10 series (Eq. 4 with γ=0);
  - `time`: seconds.

**`forecast.csv`**: Table 3, 760 rows.
- 76 datasets × 2 initialisations × (Euclidean net + soft-DTW with 4 gammas).
- `euclid` and `dtw` are the test errors on the last 40% of each series: Euclidean (sum of squares per series) and hard DTW, both averaged over the test series.
- The `Euclidean` row is the same model for both `init` values. It is duplicated for convenience.

**`classification.csv`**: Fig. 7, 711 rows.
- 79 datasets × 3 random splits (`split`) × 3 methods:
  - soft-DTW, with γ selected on validation;
  - DBA;
  - `Euclidean NC`: Euclidean-mean centroids with the Euclidean distance. It is an extra baseline that the paper does not report.
- `acc` is the test accuracy.

**`classification_all_gammas.csv`**: the same, with **all 15 gammas** of soft-DTW.
- Each γ has `val_acc` (validation) and `acc` (test).
- `classification.csv` keeps the γ with the best `val_acc`, taking the smallest γ on ties.

**`barycenters_omega.csv`**: barycenters with our variant, 20,540 rows.
- It uses exactly the same draws and initialisations as `barycenters.csv`.
- Methods:
  - DBA;
  - soft-DTW;
  - soft-DTW_omega with ω ∈ {0.1, 1} for each γ.
- It has two losses:
  - `train_loss`: hard DTW on the 10 series used to compute the barycenter. This is the Table 1 criterion.
  - `heldout_loss`: hard DTW on up to 20 **other** series of the same class. It measures whether the barycenter generalises or only fits its 10 series.

**`forecast_omega.csv`**: forecasting with our variant, 1,976 rows.
- 76 datasets × 2 initialisations × (Euclidean net + 4 gammas × {soft-DTW, soft-DTW_omega ω=0.1, ω=1}).
- Same columns as `forecast.csv`, plus `omega` and two **timing errors**:
  - `t_max_err`: distance between the time of the maximum of the forecast and that of the true series, in % of the forecast horizon, averaged over the test series;
  - `t_min_err`: the same for the minimum.
- These measure forecasts that have the right shape in the wrong place.

**`classification_omega_all.csv`**: nearest centroid with our variant, 2,370 rows.
- 79 datasets × 3 splits × 5 gammas × 2 omegas.
- `val_acc` and `acc` as in `classification_all_gammas.csv`.
- The 5 gammas are 0.001, 0.0139, 0.1, 0.72 and 10: 5 of the 15 values used for soft-DTW, as far apart as possible, so soft-DTW can be compared on exactly the same grid.

**`classification_omega.csv`**: the (γ, ω) pair selected on validation for each dataset and split. It is the best `val_acc`, with the smallest γ and then the smallest ω on ties.

**`kmeans_subset.csv`**: k-means, 1,148 rows.
- 41 datasets × 2 initialisations × 14 methods:
  - soft-DTW (4 gammas);
  - DBA;
  - subgradient;
  - soft-DTW_omega (4 gammas × 2 omegas).
- Columns:
  - `energy`: hard-DTW k-means energy (paper Eq. 5; lower is better);
  - `ari`, `nmi`: adjusted Rand index and normalised mutual information between the clusters and the true classes (higher is better; not in the paper);
  - `time`: seconds.

The scripts also checkpoint every job in `*_parts/` folders, which are not versioned.

### `summary/`: computed tables

| File | Contents |
|---|---|
| `table1_barycenters_vs_paper.csv` | % of datasets where soft-DTW has a lower mean loss than DBA / subgradient, ours vs paper |
| `table3_forecast_vs_paper.csv` | Average rank (1 = best) of each model, ours vs paper, plus the median error relative to the Euclidean net |
| `fig7_classification_by_dataset.csv` | Mean accuracy over the 3 splits per dataset and method, and the selected gammas |
| `barycenters_omega_summary.csv` | soft-DTW_omega vs DBA and vs soft-DTW, on the train and held-out losses |
| `table2_kmeans_subset_vs_paper.csv` | % of datasets where soft-DTW has a lower k-means energy than DBA / subgradient, ours (41 datasets) vs paper |
| `forecast_omega_ranks.csv` | Table 3 ranks when soft-DTW_omega replaces soft-DTW at the same γ |
| `forecast_omega_vs_softdtw.csv` | soft-DTW_omega vs soft-DTW at the same γ: DTW / Euclidean errors, timing errors, failed trainings |
| `forecast_omega_vs_euclidean_net.csv` | soft-DTW and soft-DTW_omega networks vs the Euclidean net, on both metrics at once |
| `classification_omega_by_dataset.csv` | Mean accuracy per dataset: soft-DTW (15 and 5 gammas), soft-DTW_omega, DBA, Euclidean NC |
| `classification_omega_summary.csv` | Each method vs soft-DTW on the same 5 gammas, with a Wilcoxon test |
| `kmeans_subset_omega_summary.csv` | soft-DTW_omega vs soft-DTW (same γ) and DBA: energy, ARI, NMI, Wilcoxon test on ARI |

The Fig. 7 plot on the 79 datasets is `figures/nearest_centroid_full.png`.

---

## 2. Agreement with the paper

### 2.1 Barycenters (Table 1): very close

Percentage of datasets where soft-DTW reaches a lower DTW loss (ours / paper):

| Compared with | init | γ=1 | γ=0.1 | γ=0.01 | γ=0.001 |
|---|---|---|---|---|---|
| DBA | random | 41.8 / 41 | 94.9 / 94 | 100 / 100 | 97.5 / 97 |
| DBA | euclidean | 3.8 / 4 | 44.3 / 47 | 78.5 / 80 | 96.2 / 90 |
| subgradient | random | 93.7 / 96 | 100 / 97 | 100 / 97 | 100 / 97 |
| subgradient | euclidean | 26.6 / 35 | 74.7 / 72 | 92.4 / 92 | 97.5 / 97 |

Against DBA from a random start, the match is almost exact. All the trends hold:
- small γ beats DBA almost always;
- γ=1 rarely does;
- the gap is larger from a random start.

**Differences and their explanation:**
- **Subgradient, Euclidean init, γ=1: 26.6 vs 35.** The paper does not specify the subgradient step. We use backtracking (Schultz & Jain 2018), and the method is very sensitive to the step.
- **DBA, Euclidean init, γ=0.001: 96.2 vs 90.** Our DBA is tslearn's, not the paper's. Small implementation differences can move a few datasets.

**Magnitudes (not given in the paper):**
- Median soft-DTW/DBA loss ratio:
  - random start: 0.87 (γ=0.1), 0.86 (γ=0.01) and 0.89 (γ=0.001);
  - Euclidean start: 0.98 (γ=0.01) and 0.97 (γ=0.001). The gain here is only 2–3%.
- With γ=1 the loss is higher than DBA's: 1.07× from a random start and 1.60× from the Euclidean mean. The barycenters are smoother, and the hard-DTW metric does not reward smoothness.
- Median time per barycenter: DBA ~2 s and soft-DTW ~11–26 s. With small γ, soft-DTW is ~12× more expensive.

**Extreme values with an explanation:**
- **Subgradient from a random start:** on some datasets its loss is 100–250× DBA's (HandOutlines, NonInvasiveFetalECG, Car, Fish). The step does not adapt to long series, and the method does not converge in 100 iterations. This is a weakness of the baseline, not a bug, but it inflates the win rates against it.
- **OliveOil from the Euclidean mean:** soft-DTW's loss is 3.7× DBA's. Both are essentially zero, though (DBA 0.00003, soft-DTW 0.0002). Within a class, OliveOil series are almost identical, so the ratio of two tiny numbers has no practical meaning.
- **InlineSkate:** the paper marks it N/A for some methods. We have values for all methods, which can account for a one-dataset difference in some percentages.

### 2.2 Forecasting (Table 3): same trends, one clear difference

Average rank (1 = best; ours / paper):

| Metric | init | Euclidean net | γ=1 | γ=0.1 | γ=0.01 | γ=0.001 |
|---|---|---|---|---|---|---|
| DTW | random | 3.66 / 3.46 | 3.88 / 3.55 | 2.53 / 3.33 | 2.36 / 2.79 | 2.58 / 1.87 |
| DTW | euclidean | 3.61 / 4.21 | 4.11 / 3.96 | 3.13 / 3.42 | 2.30 / 2.12 | 1.86 / 1.29 |
| Euclidean | random | 1.37 / 1.05 | 2.21 / 2.41 | 3.49 / 3.42 | 3.89 / 4.13 | 4.04 / 3.99 |
| Euclidean | euclidean | 1.99 / 1.70 | 2.96 / 2.99 | 3.01 / 3.38 | 3.22 / 3.64 | 3.82 / 3.29 |

The main claim of the paper is reproduced:
- **evaluated with DTW**, networks trained with soft-DTW and a small γ rank best;
- **evaluated with the Euclidean distance**, the Euclidean network ranks best.

From the Euclidean network, the smaller γ, the better the DTW rank: 4.11 → 3.13 → 2.30 → 1.86, as in the paper. For γ ≤ 0.01 the median test DTW is ~20% lower than the Euclidean network's.

**Main difference: γ=0.001 from random weights (rank 2.58 vs 1.87).** Training is unstable on some datasets. With γ=0.001, random init, and compared with the Euclidean network:
- the test DTW is more than 1.5× higher on 18 of the 76 datasets;
- extreme cases: Wine 16×, OliveOil 20×, Beef 3.8×, Coffee 3.2×.

These datasets have very few training series (30–60). The same instability is reported by Zeitler et al. (2023). Starting from the Euclidean network removes it, which is why that initialisation is closer to the paper.

**Huge Euclidean errors from random weights:**
- On Beef, OliveOil and Wine, soft-DTW networks trained from scratch have Euclidean errors of 100–370, while the Euclidean network's are ~0.03–1.6. Their DTW errors stay moderate.
- soft-DTW does not penalise where in time the shape falls. With so little data, the network learns forecasts with the right shape but shifted or degenerate.
- This is not a numerical error but the weakness of the method that the report already discusses. It is also why the median Euclidean error relative to the Euclidean network is 1.3–2.4× from random weights, but only 1.05–1.13× from the Euclidean network.

**Protocol details not given in the paper:**
- We use an MLP with one hidden layer of 256 sigmoid units, trained with Adam (lr 1e-3) for 400 epochs.
- The report says "79 datasets" for the paper's Table 3. Appendices K/L actually list **76**, which are the ones used here.

### 2.3 Nearest-centroid classification (Fig. 7): more favourable to soft-DTW than in the paper

- **soft-DTW is at least as accurate as DBA on 88.6% of the datasets** (70 of 79, including 2 ties). The paper reports **75%**.
- Mean accuracy over the 79 datasets:
  - soft-DTW: 0.723;
  - DBA: 0.671;
  - Euclidean centroid: 0.606.
- The median gain of soft-DTW over DBA is +3.4 points.
- soft-DTW's losses are small; the worst is BeetleFly (−3.3 points). Its gains can be large: GunPoint +24, Fish +23.5, ShapeletSim +23, ECGFiveDays +21.
- The γ selected on validation spread over the whole range [0.001, 10]. No single value dominates, so validating γ is necessary.

**Differences and their explanation:**
- The paper **does not publish the dataset list** for Fig. 7. We use the barycenter list (79). If the paper's list differs, the percentages are not directly comparable.
- The 50/25/25 split is random. With 3 splits per dataset there is noise, especially on small datasets.
- **Phoneme:** one class has so few examples that the second split (validation/test) cannot be stratified. In that case only, the second split is unstratified. The first split (train / rest) is always stratified.

### 2.4 k-means (Table 2): same trend for small γ, γ=1 much weaker than in the paper

**Subset.** We ran the 41 cheapest of the paper's 57 k-means datasets (cost ~ n_train^1.3 × length²: everything up to the Cricket datasets). The 16 left out (plus HandOutlines, used by the paper only with the Euclidean initialisation) are the long series (500–2,700 points) and the very large training sets (FordA/B, ElectricDevices, StarLightCurves…). Calibrated on the real run times of the 41 datasets (152 CPU hours), these 17 would take ~4,200 CPU hours, ~96% of the cost of the full Table 2. The protocol is the paper's:
- whole training split;
- k = number of classes;
- at most 30 Lloyd and 100 barycenter iterations;
- one random initialisation (k random series) and one Euclidean k-means initialisation.

All 14 methods of a (dataset, init) start from the same centroids. The run took 7h40 (1,148 jobs); the longest job (CricketX, soft-DTW γ=0.01) took 2 h.

Percentage of datasets where soft-DTW reaches a lower k-means energy (ours on 41 datasets / paper on 57–58):

| Compared with | init | γ=1 | γ=0.1 | γ=0.01 | γ=0.001 |
|---|---|---|---|---|---|
| DBA | random | 0 / 16 | 31.7 / 25 | 61.0 / 60 | 85.4 / 77 |
| DBA | euclidean | 2.4 / 29 | 24.4 / 24 | 58.5 / 55 | 85.4 / 69 |
| subgradient | random | 9.8 / 42 | 53.7 / 58 | 80.5 / 76 | 87.8 / 96 |
| subgradient | euclidean | 9.8 / 46 | 31.7 / 50 | 75.6 / 66 | 97.6 / 84 |

- **For γ ≤ 0.01 we reproduce the paper:** the smaller γ, the more often soft-DTW beats DBA and the subgradient method. Against DBA at γ=0.01 the numbers are almost identical (61 / 60 and 58.5 / 55).
- **The gains are small:** the median soft-DTW/DBA energy ratio is 0.99 (γ=0.01) and 0.98 (γ=0.001) from either initialisation.
- **Clear difference at γ=1:** soft-DTW almost never beats DBA (0–2.4% vs 16–29% in the paper). Its energy is ~1.5× DBA's: the γ=1 centroids are smooth, and the hard-DTW energy does not reward smoothness.
  - We have no definitive explanation for why the paper's γ=1 numbers are higher.
  - Our subset contains the shorter, cheaper datasets, which may behave differently.
  - The paper's percentages are over 57–58 datasets, so they are not strictly comparable with ours.

---

## 3. Our variant: soft-DTW_omega

### 3.1 Barycenters

We use the same 79 datasets, draws and initialisations as Table 1, with ω ∈ {0.1, 1}.

**Integrity check:** the 7,900 soft-DTW and DBA rows of this run are **bit-identical** to those in `barycenters.csv`. This confirms that the optimised code gives exactly the same numbers.

**Results** (median ratio of held-out losses, soft-DTW_omega / soft-DTW with the same γ; <1 means the variant is better):

| init | γ | ω=0.1 | ω=1 |
|---|---|---|---|
| random | 1 | 0.999 | 1.005 |
| random | 0.1 | 1.003 | 1.114 |
| random | 0.01 | 0.999 | 1.088 |
| random | 0.001 | 0.994 | 1.072 |
| euclidean | 0.01 | 1.065 | 1.247 |
| euclidean | 0.001 | 1.069 | 1.263 |

- With **ω=0.1 from a random start**, the variant **ties** with soft-DTW: it wins on 43–52% of the datasets depending on γ.
- With **ω=1**, or from the Euclidean mean, it falls behind.
- The variant is **faster for γ ≤ 0.1**. Median time per barycenter:
  - ω=0.1: 8–10 s;
  - ω=1: 3.3–3.6 s;
  - soft-DTW: 13–16 s.

  The penalty removes many degenerate paths. At γ=1 the times are similar (5.5–7.4 s vs 6.9 s).

**Caveat when reading these numbers:** the evaluation uses DTW **without** a penalty, which is exactly what soft-DTW optimises as γ → 0. The comparison therefore favours soft-DTW by construction, so a tie for ω=0.1 is a reasonable outcome.

**Comparison with the reduced report (12 datasets):** the report states that soft-DTW_omega (γ=ω=0.1, random init) has an 8% lower loss than soft-DTW. **This does not hold on the 79 datasets**: the median of the same ratio is 1.002, a tie. The report's figure came from the small dataset selection and needs correcting.

**Expected NaNs:** `heldout_loss` has 442 NaN.
- They come from 34 of the 1,580 cases, in FiftyWords, OliveOil and Phoneme.
- In those classes no series is left over after drawing the 10, so there is nothing to hold out.
- They affect all 13 methods of a case at once, so they do not bias the comparison. These cases are excluded from the held-out comparisons.

### 3.2 Forecasting: the strongest result of the variant

**Integrity check:** the 760 Euclidean and soft-DTW rows of `forecast_omega.csv` are **identical** to `forecast.csv`.

**The key comparison: against the Euclidean network.** The paper's dilemma is that no network wins on both metrics:
- the Euclidean network places the forecast correctly in time, but distorts its shape;
- soft-DTW gets the shape right, but places it in the wrong place, and sometimes training fails.

The variant changes this. Each network vs the Euclidean network, from random weights, γ=0.01 (medians over the 76 datasets; γ=0.001 gives the same picture):

| Network | DTW error vs Euclidean net | Euclidean error vs Euclidean net | Datasets where it beats the Euclidean net on **both** metrics |
|---|---|---|---|
| soft-DTW | **−19%** | **2.2× worse** | 7% |
| soft-DTW_omega, ω=0.1 | −13% | 1.4× worse | 11% |
| **soft-DTW_omega, ω=1** | **−6%** | **same (1.01×)** | **28%** |

**With ω=1 the network matches the Euclidean network's Euclidean error and still has a lower DTW error** (better DTW on 75–76% of the datasets). It gets the shape benefit of soft-DTW without giving up the timing. This compromise does not exist with plain soft-DTW.

The same holds from the Euclidean network: the share of datasets where the network beats the Euclidean net on both metrics rises from 9–20% (soft-DTW, γ ≤ 0.01) to 38–43% (variant), with an Euclidean error equal to the net's (1.01–1.02×) and a DTW error 11–16% lower.

Table: `summary/forecast_omega_vs_euclidean_net.csv`.

**Compared with soft-DTW at the same γ:**

From random weights, medians over the 76 datasets:

| γ | ω | DTW error | Euclidean error | Datasets where the Euclidean error is >10× the Euclidean net's |
|---|---|---|---|---|
| 0.01 | 0 (soft-DTW) | – | – | 22 |
| 0.01 | 0.1 | +0.7% | −14% | 9 |
| 0.01 | 1 | +8% | **−52%** | **1** |
| 0.001 | 0 (soft-DTW) | – | – | 24 |
| 0.001 | 0.1 | +0.4% | −19% | 9 |
| 0.001 | 1 | +8% | **−54%** | **1** |

- **ω=1 removes the failed trainings described in Section 2.2.** At γ=0.001 soft-DTW's test DTW is >1.5× the Euclidean net's on 18 datasets; with ω=1 this is left on a single dataset (Beef). Wine goes from 16× to 0.85× and OliveOil from 20× to 1.0×.
- **ω=1 divides the Euclidean error by ~2** for γ ≤ 0.01. For γ ≤ 0.1 it lowers the error on 87–91% of the datasets, at the cost of +6–8% in DTW.
- **ω=1 forecasts are better placed in time.** The median timing error of the maximum drops from 19.1% to 13.7% of the horizon (γ=0.001), slightly better than the Euclidean net (13.9%). The minimum drops from 24.8% to 17.8% (Euclidean net 19.2%).
- **ω=0.1 is almost free in DTW** (median +0.4–1.2%) and lowers the Euclidean error by 14–19% for γ ≤ 0.01. At γ=0.001 it also improves soft-DTW's DTW rank in Table 3, from 2.58 to 2.03, closer to the paper's 1.87.
- **Phoneme** has very large Euclidean errors (~650–730) with every model, the Euclidean net included. It is a hard dataset, not a problem of the variant.

**From the Euclidean network,** that initialisation already avoids the failures, so against soft-DTW the changes are small: the DTW error rises by up to 6% and the Euclidean error barely changes. The gain is in the balance between the two metrics described above.

**What the variant does not do:**
- it does not fix Beef;
- if you only care about the DTW error, plain soft-DTW is still slightly better.

Compared with the reduced report (13 datasets), the conclusions hold on the 76 datasets, with slightly smaller numbers:
- the report says ω=1 divides the Euclidean error by 2–3 on all 13 datasets; on the 76 it is ~2×, lowering it on 87–88% of them (γ ≤ 0.01);
- the report says ω=0.1 lowers it by 30%; on the 76 it is 14–19%.

### 3.3 Classification: no difference with soft-DTW

To compare fairly, soft-DTW is restricted to the **same 5 gammas** (taken from `classification_all_gammas.csv`), with γ selected on validation in the same way. Mean test accuracy over the 79 datasets:

| Method | Mean accuracy | vs soft-DTW, 5 gammas (better / tied / worse) | Wilcoxon p |
|---|---|---|---|
| soft-DTW, 15 gammas | 0.7229 | 51.9 / 21.5 / 26.6 % | 0.016 |
| **soft-DTW, 5 gammas** | **0.7218** | – | – |
| soft-DTW_omega, 5 γ × 2 ω | 0.7213 | 55.7 / 3.8 / 40.5 % | 0.50 |
| soft-DTW_omega, ω=0.1 only | 0.7175 | 49.4 / 8.9 / 41.8 % | 0.91 |
| soft-DTW_omega, ω=1 only | 0.7105 | 40.5 / 5.1 / 54.4 % | 0.06 |
| DBA | 0.6709 | 10.1 / 6.3 / 83.5 % | <0.001 |
| Euclidean NC | 0.6061 | 12.7 / 1.3 / 86.1 % | <0.001 |

- **soft-DTW_omega and soft-DTW are statistically indistinguishable** (mean difference −0.05 points, p = 0.50). This confirms the reduced report, which found no improvement either.
- The variant still beats DBA clearly: it is at least as accurate on 78.5% of the datasets (soft-DTW: 89.9%).
- **Large swings on individual datasets, in both directions:**
  - losses: BeetleFly −16.7 points, Trace −12.7, GunPoint −12.0;
  - gains: CinCECGTorso +13.1, Wafer +12.4, Car +8.9.
  - In Trace and GunPoint the classes differ partly by when events happen, so a warping penalty plausibly matters there; we have not verified this.
- ω=0.1 is selected on validation in 63% of the cases.
- The variant is cheaper: the median job takes 129 s (ω=0.1) and 88 s (ω=1), vs 205 s for soft-DTW.
- Using all 15 gammas instead of 5 improves soft-DTW only slightly (+0.1 points, p = 0.016), so restricting to 5 gammas costs little.

### 3.4 k-means: worse on the paper's energy, slightly better at finding the classes

Same 41 datasets and initial centroids as Section 2.4.

The k-means **energy** is the paper's score: the sum, over all series, of the hard-DTW distance between each series and the centroid of its cluster (lower = more compact clusters). It is measured with plain DTW, without a penalty.

**On the hard-DTW energy (the paper's criterion), soft-DTW_omega loses to soft-DTW:**
- for γ ≤ 0.01 it has a lower energy on only 15–17% of the datasets with ω=0.1 (median energy +5–7%), and on 2–10% with ω=1 (+24–36%);
- at γ ≥ 0.1 with ω=0.1 they are tied.

As with barycenters, this criterion favours soft-DTW by construction.

**On recovering the true classes (ARI), ω=0.1 is slightly better.** Mean ARI over the 41 datasets:

| init | γ | soft-DTW | ω=0.1 | ω=1 | DBA |
|---|---|---|---|---|---|
| random | 0.1 | 0.252 | 0.262 | 0.234 | 0.223 |
| random | 0.01 | 0.231 | 0.254 | 0.227 | 0.223 |
| random | 0.001 | 0.220 | 0.245 | 0.226 | 0.223 |
| euclidean | 0.1 | 0.309 | **0.326** (p=0.012) | 0.275 | 0.285 |
| euclidean | 0.01 | 0.282 | **0.314** (p=0.005) | 0.275 | 0.285 |
| euclidean | 0.001 | 0.282 | **0.307** (p=0.032) | 0.275 | 0.285 |

(p: Wilcoxon test of ω=0.1 vs soft-DTW at the same γ.)

- **ω=0.1 gives the highest mean ARI of all methods from both initialisations** (0.262 random, 0.326 Euclidean).
- From the Euclidean initialisation the gain over soft-DTW is statistically significant for γ ≤ 0.1 (+1.7 to +3.3 ARI points).
  - Caveat: we ran 16 such tests. With a strict Bonferroni correction (threshold 0.05/16 ≈ 0.003) none survives; γ=0.01 (p = 0.005) comes closest. Treat this as a consistent trend rather than a proven effect.
- From a random initialisation the gain is similar in size (+1 to +2.5 points) but not significant (p = 0.08–0.90).
- ω=1 does not help.
- In absolute terms all ARIs are low (0.2–0.33): with k = number of classes, k-means recovers the UCR classes poorly whatever the method.
- NMI gives the same picture (`kmeans_subset_omega_summary.csv`).
- Time: the variant is again faster. Median job time is 32–52 s for γ ≤ 0.01, vs 97–124 s for soft-DTW.

### 3.5 Overall assessment of the variant

| Experiment | soft-DTW_omega vs soft-DTW |
|---|---|
| Barycenters | Tie with ω=0.1 from a random start; worse with ω=1 or from the Euclidean mean (on DTW losses, which favour soft-DTW) |
| Forecasting | **Clearly better.** With ω=1 the network matches the Euclidean net's Euclidean error while keeping a lower DTW error, which neither soft-DTW nor the Euclidean net achieves. It also removes the failed trainings (24 → 1 datasets) and places forecasts better in time than the Euclidean net itself |
| Classification | No significant difference |
| k-means | Worse on the DTW energy; slightly better ARI with ω=0.1 (significant from the Euclidean init) |
| Cost | 1.5–4.5× faster than soft-DTW for small γ in barycenters, classification and k-means (time not measured in forecasting) |

The main result is in forecasting: **the penalty turns soft-DTW, a loss specialised in shape, into a balanced loss that keeps the timing, and it removes the failed trainings.** Elsewhere the variant is roughly neutral on the paper's metrics (which are hard DTW and favour soft-DTW by construction), slightly better at recovering the classes in k-means, and cheaper to compute.

---

## 4. NaNs and empty values: all explained

| File | Column | Why |
|---|---|---|
| all | empty `gamma` | Methods without γ: DBA, subgradient, Euclidean net, Euclidean NC |
| `barycenters_omega.csv`, `forecast_omega.csv`, `kmeans_subset.csv` | empty `omega` | Rows of the other methods (soft-DTW is ω=0) |
| `classification_all_gammas.csv` | empty `val_acc` | DBA and Euclidean NC have no hyperparameter, so there is no validation |
| `barycenters_omega.csv` | empty `heldout_loss` (442) | Classes with no spare series for the held-out set (Section 3) |

`kmeans_subset.csv` has some **negative ARI** values (minimum −0.056). These are valid: the clusters agree with the classes slightly less than chance would.

There are no infinite values and no failed runs. Every job finished:
- barycenters: 790 jobs;
- forecasting: 76;
- classification: 3,792;
- omega barycenters: 790;
- omega forecasting: 76;
- omega classification: 2,370;
- k-means: 1,148.

## 5. Reproducibility notes

- **The `time` columns are wall-clock times with 20 parallel processes on one laptop.** Use them to compare methods within an experiment, not across experiments.
- **Forecasting depends on the number of PyTorch threads,** because the floating-point summation order changes. The results reproduce bit for bit with the same thread setting:
  - 3 threads per process for normal datasets;
  - all threads for the largest ones, which run alone because of memory.
- **The reduced results in `results/` were produced with a different scipy version.** That is why some numbers differ slightly from this run, although the code is the same.
