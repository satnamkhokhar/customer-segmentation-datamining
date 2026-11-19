
# Comparative Analysis of Clustering Algorithms for Customer Segmentation Using Purchasing Behavior

**Course:** CSC 4740 — Data Mining  
**Team Members:** Satnam Singh Khokhar and Shabib Hyder Gazi  
**GitHub Repository:** https://github.com/satnamkhokhar/customer-segmentation-datamining

---

## 1. Project Overview

Customer segmentation is an important application of data mining that helps businesses understand purchasing behavior by grouping customers with similar characteristics.

This project investigates and compares three unsupervised machine learning algorithms:

1. K-Means Clustering
2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
3. Agglomerative Hierarchical Clustering

We use the UCI Online Retail dataset and construct customer-level Recency, Frequency, and Monetary Value (RFM) features.

Rather than determining the best clustering algorithm using a single evaluation metric, our research examines clustering quality, stability, customer coverage, preprocessing sensitivity, computational performance, and customer segment interpretability.

### Research Question

**Which clustering algorithm produces the most meaningful, well-defined, and stable customer segments based on purchasing behavior?**

### Original Technical Contribution

Our main technical contribution is the **Quality–Stability–Coverage Score (QSCS)**, a proposed composite evaluation method combining clustering quality, assignment stability, and customer coverage.

We also evaluate the sensitivity of this score to different weighting configurations.

QSCS is a project-specific experimental method rather than a universally validated clustering metric.

---

## 2. Research Objectives

The main objectives of this project are:

1. Clean and preprocess real-world retail transaction data.
2. Construct customer-level RFM features.
3. Implement K-Means, DBSCAN, and Hierarchical Clustering.
4. Compare clustering performance using established evaluation metrics.
5. Investigate cluster stability through repeated subsampling.
6. Examine the effects of different preprocessing techniques.
7. Develop and evaluate the proposed QSCS metric.
8. Interpret customer segments using their purchasing characteristics.
9. Build an interactive application to demonstrate clustering experiments.
10. Provide reproducible experiments, automated tests, and saved research results.

---

## 3. Dataset

### 3.1 Data Source

**Dataset:** Online Retail  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/352/online+retail  
**DOI:** https://doi.org/10.24432/C5BW33

The dataset contains transactions from a UK-based online retailer between December 2010 and December 2011.

### 3.2 Dataset Summary

| Property | Value |
|---|---|
| Original transaction records | 541,909 |
| Cleaned valid purchase records | 392,692 |
| Removed records | 149,217 |
| Unique customers | 4,338 |
| Original dataset attributes | 8 |
| Customer-level analytical features | 3 (RFM) |
| Currency | GBP (£) |
| RFM reference date | December 10, 2011 |

The original dataset contains:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

### 3.3 Data Cleaning

The preprocessing pipeline handles:

- Missing customer identifiers
- Exact duplicate records
- Canceled invoices
- Nonpositive quantities
- Invalid or nonpositive unit prices
- Invalid transaction dates
- Missing invoice identifiers

After preprocessing, 392,692 valid transactions remain, representing 4,338 customers.

The cleaning process is documented in:

`results/final/cleaning_summary.csv`

Canceled transactions and returns are excluded from the primary analysis. Therefore, Monetary Value represents positive purchase spending rather than net revenue after refunds.

---

## 4. RFM Feature Engineering

We represent each customer using three purchasing behavior features.

### 4.1 Recency (R)

Recency measures the number of calendar days since a customer's most recent purchase.

The reference date is December 10, 2011, one day after the latest valid transaction.

**Lower Recency indicates more recent purchasing activity.**

### 4.2 Frequency (F)

Frequency measures the number of distinct purchase invoices associated with a customer.

**Higher Frequency indicates more frequent purchasing activity.**

### 4.3 Monetary Value (M)

Monetary Value represents the total positive purchasing value of a customer.

Monetary = Sum(Quantity × UnitPrice)

**Higher Monetary Value indicates greater observed spending.**

### 4.4 Feature Transformation

The baseline preprocessing configuration uses:

- Original Recency values
- `log1p(Frequency)`
- `log1p(Monetary)`
- StandardScaler normalization

The logarithmic transformations reduce the influence of skewed purchasing distributions.

StandardScaler places the features on comparable numerical scales.

Clustering is performed using all three transformed RFM features.

Principal Component Analysis (PCA) is used only for two-dimensional visualization.

---

## 5. Clustering Algorithms

### 5.1 K-Means Clustering

K-Means is a centroid-based clustering algorithm that partitions observations into a predefined number of clusters.

It attempts to minimize within-cluster variation by iteratively updating cluster assignments and centroids.

**Advantages:**
- Computationally efficient
- Easy to interpret
- Produces complete cluster assignments
- Suitable for identifying relatively compact customer groups

**Limitations:**
- Requires specifying the number of clusters
- Sensitive to feature scaling and outliers
- Assumes relatively compact cluster structures

Our final selected K-Means model uses:

**k = 3 clusters**

### 5.2 DBSCAN

DBSCAN is a density-based clustering algorithm that groups observations according to neighborhood density.

Its main parameters are:

- `eps`: Neighborhood radius
- `min_samples`: Minimum number of samples required for a dense neighborhood

**Advantages:**
- Does not require specifying the number of clusters
- Can detect irregularly shaped groups
- Identifies unusual observations as noise

**Limitations:**
- Sensitive to parameter selection
- Sensitive to feature scaling
- May leave customers unassigned

Our final selected DBSCAN configuration uses:

- `eps = 0.5911`
- `min_samples = 4`

It identifies two clusters and 24 noise observations.

### 5.3 Agglomerative Hierarchical Clustering

Agglomerative Hierarchical Clustering progressively merges observations into larger groups based on similarity.

The implementation supports:

- Ward linkage
- Complete linkage
- Average linkage
- Single linkage

The final parameter search evaluates Ward, complete, and average linkage.

A dendrogram is generated to illustrate hierarchical merging relationships.

**Advantages:**
- Provides a hierarchical perspective on similarity
- Supports multiple linkage strategies
- Does not require random centroid initialization

**Limitations:**
- Can be computationally expensive
- Sensitive to linkage selection
- Can produce highly unbalanced clusters

Our final selected Hierarchical configuration uses:

- `k = 2`
- `linkage = average`

---

## 6. Evaluation Methodology

We evaluate clustering algorithms using several complementary metrics.

### 6.1 Silhouette Score

The Silhouette Score measures how similar an observation is to its own cluster compared with neighboring clusters.

Higher values generally indicate better cluster separation.

For computational efficiency, our final experiments use a deterministic, cluster-aware sample of approximately 300 assigned customers for Silhouette evaluation.

### 6.2 Davies–Bouldin Index

The Davies–Bouldin Index evaluates cluster similarity using within-cluster dispersion and between-cluster separation.

Lower values indicate better clustering.

### 6.3 Calinski–Harabasz Index

The Calinski–Harabasz Index compares between-cluster dispersion with within-cluster dispersion.

Higher values generally indicate more distinct clusters.

### 6.4 Additional Metrics

We also evaluate:

- Number of clusters
- Number of noise observations
- Customer coverage
- Model fitting time
- Cluster sizes
- Customer RFM profiles
- Assignment stability

For DBSCAN, internal quality metrics are computed on customers assigned to non-noise clusters.

Customer coverage is reported separately.

Metrics requiring at least two valid clusters are treated as undefined when this condition is not satisfied.

---

## 7. Parameter Search and Model Selection

We evaluate multiple parameter configurations for each clustering algorithm.

### K-Means

The final comparative parameter search evaluates:

`k = 2, 3, 4, 5, 6`

The original mid-project pipeline also explored k values from 2 through 10.

### DBSCAN

We evaluate combinations of:

- Neighborhood radius (`eps`)
- Minimum samples (`min_samples`)

Candidate epsilon values are derived from nearest-neighbor distance distributions.

### Hierarchical Clustering

We evaluate:

- Cluster counts from 2 through 4
- Ward, complete, and average linkage

The selected configuration for each algorithm is determined using a ranking based on Silhouette Score with a customer coverage penalty.

For DBSCAN, candidate configurations must also satisfy a minimum coverage threshold of 80%.

The complete parameter search results are saved in:

`results/final/parameter_search.csv`

Selected configurations are saved in:

`results/final/algorithm_comparison.csv`

---

## 8. Cluster Stability Analysis

A clustering solution should ideally remain reasonably consistent when the dataset changes.

To evaluate this property, we conduct repeated customer subsampling experiments.

### Experimental Setup

- Number of trials: 12 per algorithm
- Subsample fraction: 80%
- Base random seed: 42
- Stability metric: Adjusted Rand Index (ARI)

For each trial:

1. Select 80% of customers using a controlled random seed.
2. Fit the selected clustering algorithm on the subset.
3. Compare the subset assignments with the original model's assignments for the same customers.
4. Calculate ARI.
5. Aggregate the results across all trials.

ARI is invariant to the numerical names assigned to clusters.

A value close to 1 indicates strong agreement, while a value near 0 indicates agreement close to chance.

For DBSCAN, assigned-customer ARI excludes observations classified as noise in either compared solution. Coverage overlap and noise agreement are recorded separately.

### Stability Results

| Algorithm | Mean ARI | Valid Trials |
|---|---:|---:|
| K-Means | 0.9887 | 12 |
| DBSCAN | 0.9996 | 12 |
| Hierarchical | 0.0965 | 12 |

K-Means and DBSCAN demonstrate strong stability under the selected experimental procedure.

Hierarchical Clustering demonstrates substantially lower stability.

Results are saved in:

- `results/final/stability_trials.csv`
- `results/final/stability_summary.csv`

---

## 9. Original Technical Contribution: QSCS

### 9.1 Motivation

Traditional clustering evaluation frequently emphasizes internal cluster quality.

However, a clustering solution may achieve strong separation while producing unstable assignments or excluding a significant proportion of customers.

To investigate this issue, we propose the **Quality–Stability–Coverage Score (QSCS)**.

### 9.2 QSCS Formula

QSCS = 0.45Q + 0.35S + 0.20C

Where:

- **Q:** Normalized clustering quality
- **S:** Normalized assignment stability
- **C:** Customer coverage

### 9.3 Quality Component

Quality combines:

1. Silhouette Score
2. Inverted Davies–Bouldin Index
3. Calinski–Harabasz Index

Each metric is normalized using minimum and maximum values from the valid candidate model pool.

The normalized Davies–Bouldin value is inverted because lower original values are preferred.

The three normalized values are averaged to calculate Q.

### 9.4 Stability Component

The stability component is calculated using the mean ARI:

S = (Mean ARI + 1) / 2

This transforms ARI into a zero-to-one scale.

### 9.5 Coverage Component

Coverage represents the proportion of customers assigned to non-noise clusters:

C = Assigned Customers / Total Customers

### 9.6 Weight Selection

Our default weights are:

- Quality: 45%
- Stability: 35%
- Coverage: 20%

These weights represent our research priorities rather than universally optimal values.

### 9.7 QSCS Results

| Algorithm | QSCS |
|---|---:|
| K-Means | 0.9566 |
| DBSCAN | 0.8184 |
| Hierarchical | 0.6718 |

**K-Means achieves the highest overall QSCS under our default weighting configuration.**

### 9.8 QSCS Weight Sensitivity

We also evaluate alternative weighting configurations, including:

- Equal weights
- Quality-focused weights
- Stability-focused weights
- Coverage-focused weights
- Removing one component at a time

These experiments demonstrate how the relative importance of evaluation criteria affects algorithm rankings.

For example, when quality receives zero weight and stability and coverage receive greater importance, DBSCAN narrowly outranks K-Means.

Results are saved in:

- `results/final/qscs_comparison.csv`
- `results/final/qscs_weight_sensitivity.csv`

QSCS is a project-specific experimental contribution. Its normalization depends on the evaluated candidate models, and it has not been externally validated as a universal measure of business usefulness.

---

## 10. Preprocessing Sensitivity Analysis

Clustering results can be influenced by feature transformations and scaling.

We evaluate five preprocessing configurations:

1. **Baseline:** Log-transformed Frequency and Monetary with StandardScaler
2. **All Log:** Log transformation applied to all three RFM features
3. **Raw:** Original RFM features with StandardScaler
4. **Robust:** Log-transformed Frequency and Monetary with RobustScaler
5. **Winsor:** Winsorization-based preprocessing with StandardScaler

Each configuration is tested using the selected K-Means, DBSCAN, and Hierarchical Clustering models.

This produces:

**5 preprocessing configurations × 3 algorithms = 15 experiments**

We compare:

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index
- Customer coverage
- Execution time
- ARI relative to baseline assignments

The selected algorithm parameters remain fixed across preprocessing variants.

This allows us to isolate preprocessing effects rather than retuning each model for every variant.

### Main Observations

- K-Means remains relatively consistent under RobustScaler and winsorization.
- DBSCAN can produce only one usable cluster under some preprocessing choices.
- Hierarchical Clustering shows substantial assignment sensitivity to feature transformations.
- High numerical cluster separation does not necessarily imply stable or practically useful segmentation.

Results are saved in:

`results/final/feature_sensitivity.csv`

---

## 11. Final Experimental Results

The final experiments use 4,338 customer records.

### 11.1 Algorithm Comparison

| Algorithm | Clusters | Silhouette | Davies–Bouldin | Calinski–Harabasz | Coverage |
|---|---:|---:|---:|---:|---:|
| K-Means | 3 | 0.4547 | 0.8251 | 4393.74 | 100% |
| DBSCAN | 2 | 0.3202 | 1.1265 | 2421.55 | 99.45% |
| Hierarchical | 2 | 0.4898 | 0.7255 | 18.87 | 100% |

### 11.2 K-Means Customer Segments

| Cluster | Customers | Percentage | Interpretation |
|---|---:|---:|---|
| 0 | 2,037 | 46.96% | Relatively recent, lower-frequency and lower-spending customers |
| 1 | 981 | 22.61% | Less recent, lower-frequency and lower-spending customers |
| 2 | 1,320 | 30.43% | Recent, high-frequency and high-spending customers |

The selected K-Means model produces three interpretable purchasing-behavior groups.

These groups may help businesses investigate retention, engagement, and customer relationship strategies.

### 11.3 DBSCAN Customer Segments

DBSCAN identifies:

- Cluster 0: 2,824 customers
- Cluster 1: 1,490 customers
- Noise: 24 customers

The noise observations include customers with unusual purchasing patterns.

These customers should not automatically be interpreted as fraudulent or undesirable.

### 11.4 Hierarchical Customer Segments

The selected Hierarchical model identifies:

- Cluster 0: 4,334 customers
- Cluster 1: 4 customers

This highly unbalanced result illustrates an important limitation of relying exclusively on cluster separation metrics.

Although the selected Hierarchical model has the highest Silhouette Score, it provides limited differentiation across most customers and demonstrates low subsampling stability.

### 11.5 Main Research Finding

**The algorithm with the highest Silhouette Score is not necessarily the best overall customer segmentation solution.**

Hierarchical Clustering achieves the highest selected-model Silhouette Score, but its segmentation is extremely unbalanced and unstable under our subsampling procedure.

DBSCAN demonstrates strong assigned-customer stability but introduces noise handling and coverage considerations.

K-Means achieves:

- Three interpretable customer groups
- Strong stability
- Complete customer coverage
- The highest QSCS under our default weights

Therefore, K-Means offers the strongest overall balance for this dataset under our proposed evaluation framework.

This conclusion is specific to the experimental conditions and does not establish that K-Means is universally superior.

---

## 12. Project Architecture and File Structure

The project follows the pipeline:

Data Input → Data Cleaning → RFM Feature Engineering → Feature Transformation → Clustering → Evaluation → Visualization

### Main Project Structure

```text
customer-segmentation-datamining/
├── app.py
├── dashboard_logic.py
├── main.py
├── final_main.py
├── part2_stability_qscs.py
├── part3_sensitivity.py
├── part4_report.py
├── demo.py
├── requirements.txt
├── requirements-lock.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── final_research.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_final_research.py
│   ├── test_part2.py
│   ├── test_part3.py
│   ├── test_dashboard_logic.py
│   └── test_baseline_consistency.py
├── notebooks/
│   └── 01_customer_segmentation.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── tables/
    └── final/
        ├── algorithm_comparison.csv
        ├── parameter_search.csv
        ├── stability_trials.csv
        ├── stability_summary.csv
        ├── qscs_comparison.csv
        ├── qscs_weight_sensitivity.csv
        ├── feature_sensitivity.csv
        ├── kmeans_profiles.csv
        ├── dbscan_profiles.csv
        ├── hierarchical_profiles.csv
        ├── experiment_manifest.json
        └── figures/
```

### Important Files

| File | Purpose |
|---|---|
| `main.py` | Original preprocessing, RFM, and K-Means analysis |
| `final_main.py` | Final clustering comparison and research experiments |
| `part2_stability_qscs.py` | Stability and QSCS experiments |
| `part3_sensitivity.py` | Preprocessing sensitivity analysis |
| `part4_report.py` | Regenerates research figures and weight-sensitivity results |
| `app.py` | Interactive Streamlit dashboard |
| `dashboard_logic.py` | Analytical functions supporting the dashboard |
| `demo.py` | Console demonstration of saved experimental results |
| `src/` | Core implementation modules |
| `tests/` | Automated tests |
| `results/final/` | Final experimental results and figures |

---

## 13. Software Requirements and Installation

### 13.1 System Requirements

The project uses:

- Python 3.11 or later
- Python 3.12 recommended
- Windows, macOS, or Linux
- Internet connection for initial dependency installation and dataset download

The saved final research manifest records:

- Python 3.12.4
- macOS 15.7.7 on ARM64
- Random seed 42

### 13.2 Main Python Dependencies

The project uses:

- pandas
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- seaborn
- openpyxl
- Streamlit
- Plotly
- pytest
- Jupyter
- threadpoolctl

The required package versions and constraints are maintained in `requirements.txt`.

The older `requirements-lock.txt` represents an earlier environment snapshot. Use `requirements.txt` for the complete current application.

### 13.3 Clone the Repository

```bash
git clone https://github.com/satnamkhokhar/customer-segmentation-datamining.git

cd customer-segmentation-datamining
```

If using the professor's `Sources` submission package, extract it and open a terminal inside the `Sources` directory instead.

### 13.4 Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 13.5 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Run all subsequent commands from the project root directory or extracted `Sources` directory.

### 13.6 Dataset Setup

The original dataset workbook is not tracked in the public GitHub repository.

Download the Online Retail dataset from:

https://archive.ics.uci.edu/dataset/352/online+retail

Extract the workbook and place it at:

```text
data/raw/Online Retail.xlsx
```

Alternatively, use:

```bash
python main.py --download
```

The professor's final `Sources` submission package should include the workbook so the project can be executed without downloading the dataset separately.

---

## 14. Running the Project

### 14.1 Launch the Interactive Dashboard

```bash
python -m streamlit run app.py
```

The application is normally accessible at:

http://localhost:8501

The dashboard contains six sections.

**1. Overview & RFM**

Displays:

- Original and cleaned transaction counts
- Unique customer count
- Cleaning ledger
- RFM customer records
- Interactive RFM distributions

**2. Live Clustering**

Allows users to:

- Select K-Means, DBSCAN, or Hierarchical Clustering
- Choose preprocessing configurations
- Adjust model parameters
- Select customer sample size
- Fit clustering models interactively
- View clustering metrics
- Explore PCA visualizations
- Examine customer segment profiles
- Download segment profiles as CSV

**3. Algorithm Comparison**

Displays:

- Full-dataset algorithm comparison
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index
- Coverage
- Execution time
- Customer segment profiles

**4. Stability & QSCS**

Displays:

- Stability results
- Mean ARI comparisons
- QSCS component scores
- Interactive QSCS weight adjustments
- Algorithm rankings

**5. Preprocessing Sensitivity**

Displays:

- Results for five preprocessing configurations
- Comparisons across three algorithms
- ARI relative to baseline
- Clustering quality and coverage changes

**6. Code & Methodology**

Provides:

- Methodology explanation
- Implementation overview
- Reproducibility commands
- Research considerations and limitations

The Live Clustering section fits models interactively.

The comparison sections use saved full-dataset research outputs to avoid repeating expensive experiments during every dashboard interaction.

### 14.2 Run the Original Analysis

```bash
python main.py
```

This runs the original preprocessing, RFM, and K-Means pipeline.

### 14.3 Run the Full Comparative Research

```bash
python final_main.py --repeats 12
```

This performs:

- Dataset preprocessing
- RFM feature engineering
- Clustering parameter search
- Model selection
- Stability experiments
- QSCS evaluation
- Preprocessing comparisons
- Research result generation

The full experiment may take some time depending on hardware.

### 14.4 Run Stability and QSCS Experiments

```bash
python part2_stability_qscs.py --repeats 12
```

### 14.5 Run Preprocessing Sensitivity Analysis

```bash
python part3_sensitivity.py
```

### 14.6 Regenerate Research Figures

```bash
python part4_report.py
```

### 14.7 Display Saved Experimental Results

```bash
python demo.py
```

This prints:

- Selected algorithm comparison
- Stability summary
- QSCS scores
- Preprocessing experiment count
- Customer segment profiles
- Saved figures

This command does not require retraining all models.

**Note:** Rerunning experiments can overwrite saved results. Preserve copies if comparing different experiment configurations.

---

## 15. Automated Testing

The project includes automated tests covering:

- Dataset validation
- Missing and invalid data
- Duplicate transaction handling
- RFM calculations
- Feature transformations
- K-Means clustering
- Final clustering evaluation
- Stability calculations
- QSCS calculations and weight validation
- Preprocessing sensitivity
- Dashboard analytical functions
- Baseline metric consistency

Run the test suite using:

```bash
python -m pytest -q
```

The test suite includes synthetic fixtures to validate individual operations.

Research results are generated from the real UCI Online Retail dataset, not synthetic data.

Automated tests verify implementation behavior but do not establish that any particular customer segmentation is objectively correct or commercially optimal.

---

## 16. Experimental Results and Visualizations

The final research outputs are saved in:

```text
results/final/
```

### Experimental Tables

| File | Description |
|---|---|
| `parameter_search.csv` | Evaluated clustering parameter configurations |
| `algorithm_comparison.csv` | Selected algorithm performance |
| `stability_trials.csv` | Individual stability experiments |
| `stability_summary.csv` | Aggregated stability results |
| `qscs_comparison.csv` | Quality, stability, coverage, and QSCS |
| `qscs_weight_sensitivity.csv` | Alternative QSCS weighting experiments |
| `feature_sensitivity.csv` | Preprocessing sensitivity results |
| `kmeans_profiles.csv` | K-Means customer segment characteristics |
| `dbscan_profiles.csv` | DBSCAN customer segment characteristics |
| `hierarchical_profiles.csv` | Hierarchical customer segment characteristics |
| `experiment_manifest.json` | Experiment configuration and reproducibility metadata |

### Generated Figures

The `results/final/figures/` directory contains:

- `algorithm_silhouette.png`
- `algorithm_coverage.png`
- `stability.png`
- `qscs.png`
- `preprocessing_sensitivity.png`
- `qscs_weight_sensitivity.png`
- `dbscan_noise.png`
- `dendrogram.png`

These figures illustrate clustering quality, stability, coverage, preprocessing effects, QSCS results, and hierarchical relationships.

The dendrogram is generated using Ward linkage on a sampled set of customers for illustrative purposes. It does not represent the exact selected average-linkage model.

---

## 17. Research Limitations

Our study has several limitations.

1. **Historical dataset:** The Online Retail dataset represents purchasing behavior from 2010–2011 and may not reflect current retail patterns.

2. **Single retailer:** Results may not generalize to other businesses or industries.

3. **No ground-truth labels:** The dataset does not contain verified customer segment labels.

4. **No measured business outcomes:** The project evaluates segmentation properties but does not demonstrate improvements in customer retention, marketing conversion, or revenue.

5. **Excluded returns:** Monetary Value measures positive purchases rather than net revenue after refunds.

6. **DBSCAN noise handling:** Quality and assigned-customer stability metrics must be interpreted alongside coverage and noise statistics.

7. **Sampled Silhouette:** The final Silhouette metric is calculated on a deterministic sample for computational efficiency.

8. **QSCS assumptions:** The score depends on researcher-defined weights and candidate-pool normalization.

9. **Preprocessing sensitivity:** Selected hyperparameters remain fixed across preprocessing variants, so the results do not represent separately optimized models for each transformation.

10. **Hierarchical cluster imbalance:** The selected Hierarchical solution separates only four customers from the remaining population, limiting its usefulness as a broad customer segmentation.

---

## 18. Learning Outcomes

Through this project, we gained practical experience in:

- Real-world data cleaning and validation
- RFM feature engineering
- Unsupervised machine learning
- Clustering parameter selection
- Internal clustering evaluation
- Stability analysis using Adjusted Rand Index
- Designing custom evaluation metrics
- Controlled preprocessing experiments
- Customer segment interpretation
- Scientific visualization
- Interactive dashboard development
- Modular Python implementation
- Automated testing
- Reproducible experimental research

The project also demonstrated that strong numerical performance on one metric does not necessarily imply meaningful, stable, or useful customer groups.

Individual team member contributions, teamwork experiences, and group member evaluation are documented separately in the final project report.

---

## 19. References

1. Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

2. Hughes, A. M. (1994). Strategic Database Marketing. Probus Publishing.

3. Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). RFM and CLV: Using iso-value curves for customer base analysis. Journal of Marketing Research, 42(4), 415–430.

4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.

5. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. Proceedings of KDD-96.

---

## Conclusion

This project demonstrates how different clustering algorithms identify customer segments from purchasing behavior and why clustering evaluation should consider more than a single numerical metric.

Using the UCI Online Retail dataset, we implemented and evaluated K-Means, DBSCAN, and Hierarchical Clustering on 4,338 customer records.

Our experiments examined clustering quality, stability, preprocessing sensitivity, customer coverage, and segment interpretability.

We also proposed the Quality–Stability–Coverage Score (QSCS), which combines multiple evaluation dimensions into a single transparent comparison framework.

Under our default QSCS weights, **K-Means achieved the highest overall score**, producing three interpretable customer groups with strong stability and complete coverage.

The project highlights the importance of evaluating clustering algorithms through multiple complementary criteria rather than relying exclusively on cluster separation.
