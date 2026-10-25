# First progress report: customer segmentation

## 1. Project's Current Update
Executed real UCI Online Retail acquisition, exploration, sequential cleaning, RFM engineering, feature transformation and scaling, K-Means K=[2, 3, 4, 5, 6, 7, 8, 9, 10], evaluation, profiling, and figure/table exports.
Dataset SHA-256: `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`. Reference date: **2011-12-10** (one day after the latest valid purchase, using calendar-day recency).

## 2. Results Obtained
Original transaction rows: **541,909**. Cleaned rows: **392,692**. Removed: **149,217**. Customers clustered: **4,338**.

Preliminary K=3: candidates within 0.03 of the best silhouette (0.4157); prefer profiles with all segments >=5% of customers, then nearest to the normalized inertia-curve elbow K=4, breaking ties toward fewer segments. Profiles describe differences in original RFM units; labels are relative descriptions, not demonstrated churn. This transparent heuristic is provisional, not proof of optimal business usefulness.

Selected-model metrics: {"inertia": 4299.190067675818, "silhouette": 0.4156506011553123, "davies_bouldin": 0.8250669233107493, "calinski_harabasz": 4393.736066642546, "fit_seconds": 0.025537187000054473}. Fit time excludes evaluation, loading, and plotting.

```text
 Cluster  customers  mean_recency  median_recency  mean_frequency  median_frequency  mean_monetary  median_monetary  percentage                                                                           description
       0       2037         55.25           46.00            2.05              2.00         612.71           508.80       46.96   more recent, lower frequency, lower spending (relative to overall customer medians)
       1        981        255.62          253.00            1.39              1.00         397.17           290.51       22.61   less recent, lower frequency, lower spending (relative to overall customer medians)
       2       1320         30.60           18.00            9.84              7.00        5492.04          2570.49       30.43 more recent, higher frequency, higher spending (relative to overall customer medians)
```

Descriptions compare segment medians with overall customer medians. Less recent activity does not establish churn. Purchase values are in GBP; returns are excluded, so monetary values are not net revenue.

Primary configuration: log1p Frequency and Monetary, raw Recency, StandardScaler, seed 42, n_init=20; no outliers removed. Raw and transformed skewness, IQR outlier counts, and the alternate log-Recency skewness are recorded in `models/transformation.json`. Recency remains untransformed for this interpretable baseline; sensitivity experiments are future work.

See [cleaning ledger](tables/cleaning_summary.csv), [RFM statistics](tables/rfm_statistics.csv), [K comparison](tables/kmeans_comparison.csv), [profiles](tables/segment_profiles.csv), and [machine-readable results](tables/experimental_summary.json).

![RFM distributions](figures/rfm_distributions.png)
![Elbow](figures/inertia.png)
![Silhouette](figures/silhouette.png)
![Davies–Bouldin](figures/davies_bouldin.png)
![Cluster sizes](figures/cluster_sizes.png)
![PCA](figures/pca_segments.png)
![RFM profiles](figures/cluster_rfm.png)
PCA is visualization only; clustering uses all three scaled RFM features.
Additional EDA figures: `missing_values.png`, `monthly_transactions.png`, `transaction_values.png`, `top_countries.png` under `figures/`.

## 3. Progress Compared to Original Milestones
The proposal schedules exploration for October 19–25, cleaning/RFM for October 26–November 1, and algorithm work for November 2–15, 2026. This run completes exploration, cleaning/RFM, and preliminary K-Means ahead of those later planned stages. It does not complete the full algorithm-comparison milestone. Evidence is limited to the outputs of this executed pipeline; test validation is reported separately.

## 4. Future Work
Implement DBSCAN and Hierarchical Clustering; compare all three methods under comparable preprocessing; investigate sampling/seed stability using ARI on common customers; evaluate scaling/outlier sensitivity; investigate the proposed combined quality/stability score; prepare the final report and demonstration. None of these future experiments is claimed complete.
