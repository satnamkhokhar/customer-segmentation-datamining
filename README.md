# CSC 4740 — Customer Segmentation: Code & Demonstration

**Team:** Shabib Hyder Gazi and Satnam Singh Khokhar  
**Project:** Comparative Analysis of Clustering Algorithms for Customer Segmentation Using Purchasing Behavior

This is the **code-only submission**: no research report or theoretical-analysis document is included. It contains executable code, automated tests, the UCI Online Retail workbook, real-data result tables, and visualizations for a live demonstration.

## Requirements
Python 3.11+ (Python 3.12 recommended); dependencies and versions are pinned in `requirements.txt`, with a full environment snapshot in `requirements-lock.txt`. Works on Windows, macOS, and Linux with compatible Python packages. No API keys, databases, or accounts required.

## Quick demo (recommended for presentation)
Open a terminal **inside the `Sources` directory**, then:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py
python part4_report.py
```

`demo.py` immediately displays the saved real-data comparison, 12-trial stability summary, proposed QSCS rankings, segment profiles, and locations of the six figures. It does not rerun expensive model selection. `part4_report.py` regenerates **only charts and the QSCS weight-sensitivity CSV**, not a research report.

## Live code execution
The original UCI workbook is bundled at `data/raw/Online Retail.xlsx` so you can demonstrate code without internet access.

```bash
python main.py                        # original K-Means RFM baseline
python final_main.py --repeats 12     # complete 3-algorithm tuning + stability; slow
python part2_stability_qscs.py --repeats 12  # reuse saved parameter grid
python part3_sensitivity.py           # 15 preprocessing sensitivity comparisons
python -m pytest -q                  # automated tests
```

For a short live presentation, use `python demo.py` rather than rerunning the expensive full grid. Re-running scripts updates result files; keep a backup if you want to preserve the supplied results.

## Implementation and novel contribution
- Cleaning, customer-level Recency/Frequency/Monetary features, log transformations, scaling, and original K-Means baseline.
- DBSCAN and agglomerative hierarchical clustering, hyperparameter search, quality metrics (silhouette, Davies–Bouldin, Calinski–Harabasz), noise coverage, timing, and segment profiles.
- 12 seeded 80% subsampling trials with customer-ID-aligned Adjusted Rand Index (ARI); noise overlap diagnostics.
- **Project-specific QSCS** = 0.45 × normalized quality + 0.35 × normalized stability + 0.20 × coverage, with weight/component sensitivity tests. This is a proposed evaluation method, not an externally validated metric.
- Five preprocessing variants per algorithm (15 sensitivity runs); six saved PNG comparison figures.
- Automated synthetic-case tests in `tests/`, and a Jupyter notebook in `notebooks/`.

## Files and demonstration talking points
- `src/`: data processing, clustering, evaluation, and custom score.
- `final_main.py`: full research workflow; `demo.py`: fast saved-results walkthrough.
- `results/final/`: parameter grid, customer assignments, segment profiles, stability trials, QSCS and sensitivity CSVs, figures.
- `tests/`: 29 tests.
- The original mid-project K-Means chose **k=3**; the final comparison grid selected **k=2** using a different selection rule.
- DBSCAN's noise-excluded ARI may look optimistic; show coverage and noise agreement as well.
- This is unsupervised segmentation, not a validated predictor of churn or future spending.

**Data source:** UCI Machine Learning Repository, Online Retail (DOI: 10.24432/C5BW33), https://archive.ics.uci.edu/dataset/352/online+retail.

## Interactive Streamlit dashboard (recommended for the 20-minute video)

The dashboard is implemented in `app.py` and the testable calculation layer is in `dashboard_logic.py`. Install the updated requirements (including Streamlit and Plotly) and run from `Sources/`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` by default. Its six sections are **Overview & RFM**, **Live clustering**, **Algorithm comparison**, **Stability & QSCS**, **Preprocessing sensitivity**, and **Code & methodology**. In **Live clustering**, click **Fit model** to train an actual model on the selected number of customers; the other comparison views intentionally load the saved real-data experiment tables to avoid a lengthy parameter search during a recorded presentation. The live fit uses a deterministic random customer sample, while the full experiment outputs cover 4,338 customers.

**Important environment note:** the dashboard source and its calculation layer were checked, but the authoring environment could not download Streamlit from PyPI to launch the actual browser interface. Before recording, run the commands above on your machine, click through all six pages, and verify that a live fit works. This browser-level check is still required.

The `requirements-lock.txt` snapshot predates the Streamlit addition; use `requirements.txt` for the dashboard. For a fresh full lock file after installation, run `python -m pip freeze > requirements-lock.txt` in your activated environment.
