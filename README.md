# Comparative Analysis of Clustering Algorithms for Customer Segmentation Using Purchasing Behavior

CSC 4740 Data Mining research project — Shabib Hyder Gazi and Satnam Singh Khokhar.

## Objective and preliminary scope
Which clustering algorithm produces the most meaningful, well-defined, and stable customer segments based on purchasing behavior? The full proposal compares K-Means, DBSCAN, and Hierarchical Clustering. **This first progress-report implementation covers only exploration, cleaning, RFM features, and preliminary K-Means.** It does not establish algorithm superiority or cluster stability.

## Dataset
[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), Chen (2015), DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Approximately 541,909 line-item transactions, December 2010–December 2011, from a UK-based retailer. Currency: GBP (£). All countries are retained; this is not a UK-only customer analysis.

Download the official ZIP, extract `Online Retail.xlsx`, and place it at `data/raw/Online Retail.xlsx`, or use `python main.py --download`. The helper uses the official HTTPS archive and does not overwrite an existing workbook. Raw data remain unchanged. Workbook SHA-256 and software versions are recorded in the experimental summary. No research results are simulated.

## Installation (Python 3.11+)
Run from this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --download
```

`requirements.txt` pins direct dependencies from the validated environment. `requirements-lock.txt` freezes the complete Python 3.12 environment; use it for closest reproduction on Python 3.12. Platform/wheel availability may differ on other Python versions.

For the prepared cloud instance, use `/workspace/customer-segmentation-venv/bin/python` or activate `/workspace/customer-segmentation-venv/bin/activate`. No credentials or database are required. Initial acquisition requires `archive.ics.uci.edu`; installation uses PyPI and its file CDN. After download, analysis runs locally.

## Run and notebook

```bash
python main.py                       # existing local workbook
python main.py --dataset path/to/Online\ Retail.xlsx
python main.py --k 3                  # explicit researcher override; inspect comparisons first
python -m jupyter lab                # open notebooks/01_customer_segmentation.ipynb
```

The notebook calls the same `run_pipeline` function as the CLI and executes without editing paths from either the root or notebooks directory. Missing data produces instructions instead of invented output. All K candidates and their profiles are exported. A run replaces outputs at standard paths; archive them before running a different experiment. Pickle model/scaler files must only be loaded from trusted sources.

## Methods
Loading validates all eight expected columns and numeric purchase fields, parses dates, and records original quality statistics before cleaning. Invalid dates are counted and later removed. Cleaning sequentially removes missing customer IDs, exact duplicate rows, cancellation invoices (C prefix), nonpositive or nonfinite quantities/prices, invalid dates, and missing invoice IDs. The ledger reconciles removals; its percentages use the rows remaining before each step. Distinct line items on the same invoice are preserved. Monetary values represent positive purchases, **not net revenue after returns**.

RFM has one record per customer: Recency is calendar days since the last purchase relative to one day after the latest valid transaction date; Frequency counts distinct invoices; Monetary sums Quantity × UnitPrice in GBP. The exact reference date is saved. Missing, infinite, or invalid RFM values fail validation.

Primary features: untransformed Recency and log1p Frequency/Monetary, followed by StandardScaler on all three. No high-spending customers are removed or winsorized. Raw/transformed skewness, IQR outlier counts, and log-Recency skewness are recorded to inform later sensitivity analysis. Raw Recency is retained for this baseline, not claimed optimal. The scaler and settings are saved.

K-Means tests K=2–10 where sample size and distinct points allow; seed=42, n_init=20, Lloyd algorithm, two computational threads. Silhouette uses every customer in the full scaled space. Higher Silhouette and Calinski–Harabasz and lower Davies–Bouldin generally indicate stronger internal separation. Inertia measures within-cluster sum of squares; fit time excludes metric calculation and I/O.

Preliminary selection considers candidates within 0.03 of the highest silhouette, prefers profiles with no segment below 5% of customers, then picks the nearest normalized inertia-curve elbow (ties favor fewer clusters). This documented heuristic combines separation, compactness, and usable sizes; it is provisional and requires domain review. Relative segment descriptions use observed RFM medians. Inactivity is not proof of churn. PCA is used **only for the figure**; clustering uses all three selected RFM features.

## Structure and outputs

```text
main.py                         shared pipeline entry point
src/data_loader.py              official download, validation, EDA statistics
src/preprocessing.py            sequential cleaning ledger
src/rfm_analysis.py             customer RFM aggregation
src/clustering.py               transformation, K-Means, selection, profiles
src/evaluation.py               clustering metrics
src/visualization.py            report-ready 300 dpi figures
notebooks/01_customer_segmentation.ipynb
tests/test_pipeline.py          synthetic correctness fixtures only
data/raw/                      ignored original Excel workbook
data/processed/                ignored cleaned transactions, RFM, all K assignments
results/tables/                exploration JSON, cleaning ledger, statistics,
                                K comparison, all K profiles, experimental summary
results/figures/               ignored high-resolution EDA/RFM/metric/PCA/profile PNGs
results/models/                ignored selected model, scaler, transformation JSON
results/progress_summary.md    generated first progress report with actual results
```

Original data, customer-level outputs, model binaries, notebook checkpoints and figure images are ignored to avoid uploading datasets or large artifacts. Small aggregate tables and the generated progress summary remain reviewable. The report figures are available locally and can be exported separately.

## Validation and implementation status
Run `python -m pytest -q` for schema, missing-dataset handling, sequential cleaning, purchase totals, exact calendar recency, distinct-invoice frequency, monetary aggregation, unique customer rows, scaling and all K assignments. Synthetic fixtures are used only in tests. **Current instance:** official UCI acquisition and the complete real-data CLI experiment succeeded. The run loaded 541,909 transactions, retained 392,692 valid purchase rows, and clustered 4,338 customers. All K=2–10 experiments completed; the provisional selection was K=3 (silhouette approximately 0.416). All 17 automated tests passed, and all five notebook code cells executed successfully on the actual UCI dataset. See `results/progress_summary.md`, `results/tables/experimental_summary.json`, and `results/tables/validation_status.json` for measured results and verification status. Successful code execution does not complete the later research milestones.

## Future work
DBSCAN and its tuning; Agglomerative Hierarchical Clustering; a full algorithm comparison; stability under sampling and seeds using ARI on common customers; scaling/outlier sensitivity; the proposed combined quality-and-stability score; final report, presentation and optional interactive demonstration. None is implemented in this milestone.
