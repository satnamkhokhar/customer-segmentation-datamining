# Part 3: Preprocessing Sensitivity

This checkpoint adds `part3_sensitivity.py`, which reads the real UCI Online Retail Excel workbook, applies the existing cleaning/RFM pipeline, and evaluates the previously selected K-Means, DBSCAN, and Hierarchical Clustering configurations across five preprocessing variants.

## Run

From the repository root, with dependencies installed and `data/raw/Online Retail.xlsx` present:

```bash
python part3_sensitivity.py --dataset 'data/raw/Online Retail.xlsx'
python -m pytest -q
```

Output: `results/final/feature_sensitivity.csv` (15 algorithm-by-variant comparisons).

## Experimental design

The same 4,338 customers are used for every run. Selected parameters come from `results/final/parameter_search.csv` and are held fixed across transformations. ARI compares customer assignments against each algorithm's baseline; label permutations do not change ARI. For DBSCAN, ARI treats noise as an additional label, while silhouette, DB, and CH exclude noise; coverage and noise count are reported separately. A cluster-aware 300-customer sample is used for silhouette to avoid randomly omitting rare clusters. Thus this sampled silhouette is not directly interchangeable with the full-dataset silhouette in the earlier mid-project report.

**Important:** A high silhouette after raw scaling can be caused by extreme monetary values and does not, by itself, imply better customer segmentation. Interpret alongside ARI, coverage, cluster sizes and profiles. DBSCAN with fewer than two non-noise clusters has undefined quality metrics, correctly represented as missing values.

This checkpoint does not claim to have completed the full 12-trial stability experiment or final research report. The previously saved `stability_summary.csv` is preliminary.
