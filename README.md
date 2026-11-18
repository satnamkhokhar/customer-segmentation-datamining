# Comparative Analysis of Clustering Algorithms for Customer Segmentation Using Purchasing Behavior

**Course:** CSC 4740 — Data Mining  
**Institution:** Georgia State University  
**Team Members:** Satnam Singh Khokhar and Shabib Hyder Gazi

## 1. Project Overview

Customer segmentation helps businesses understand differences in purchasing behavior and identify groups of customers with similar characteristics. This project conducts a comparative analysis of three unsupervised machine learning algorithms:

- K-Means Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Agglomerative Hierarchical Clustering

The study uses customer purchasing data from the UCI Online Retail dataset and constructs Recency, Frequency, and Monetary (RFM) features for each customer.

### Research Question

**Which clustering algorithm produces the most meaningful, well-defined, and stable customer segments based on purchasing behavior?**

Rather than selecting an algorithm solely based on a single clustering metric, the project evaluates cluster quality, assignment stability, customer coverage, preprocessing sensitivity, and computational performance.

The research also proposes a project-specific evaluation method, the **Quality–Stability–Coverage Score (QSCS)**, to combine several important clustering characteristics into a single comparative score.

## 2. Research Objectives

The primary objectives are to:

1. Clean and preprocess real-world retail transaction data.
2. Construct customer-level RFM features.
3. Implement and compare three clustering algorithms.
4. Evaluate clustering quality using established internal validation metrics.
5. Investigate cluster stability through repeated customer subsampling.
6. Examine how preprocessing choices affect clustering assignments.
7. Develop and evaluate a composite Quality–Stability–Coverage Score.
8. Interpret customer segments using their original purchasing characteristics.
9. Provide reproducible experiments, automated tests, saved results, and an interactive analytical application.

## 3. Dataset

### Data Source

**Dataset:** Online Retail  
**Repository:** UCI Machine Learning Repository  
**DOI:** https://doi.org/10.24432/C5BW33  
**Dataset URL:** https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains transaction records from a UK-based online retailer covering December 2010 through December 2011.

### Dataset Summary

| Property | Value |
| --- | --- |
| Original transaction records | 541,909 |
| Cleaned purchase records | 392,692 |
| Unique customers after preprocessing | 4,338 |
| Original dataset features | 8 |
| Customer-level analytical features | 3 (RFM) |
| Currency | GBP (£) |

The original fields include InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

### Data Cleaning

The preprocessing pipeline addresses several common data-quality problems:

- Missing customer identifiers
- Duplicate transaction records
- Cancelled invoices
- Nonpositive quantities
- Invalid or nonpositive unit prices
- Missing or invalid transaction dates
- Missing invoice identifiers

The pipeline records the effects of these cleaning operations, allowing the reduction from the original dataset to the final analytical dataset to be examined.

Cancelled transactions are excluded. Consequently, the Monetary feature measures positive purchases rather than net revenue after refunds or returns.

## 4. Feature Engineering: RFM Analysis

Each customer is represented using three purchasing-behavior features.

### Recency (R)

The number of calendar days since the customer's most recent purchase, measured relative to one day after the latest valid transaction date.

Lower Recency indicates more recent purchasing activity.

### Frequency (F)

The number of distinct purchase invoices associated with a customer.

Higher Frequency indicates more frequent purchasing activity.

### Monetary (M)

The total positive purchasing value associated with a customer:

Monetary = Sum(Quantity × UnitPrice)

Higher Monetary indicates greater observed purchasing value.

### Feature Transformation

The baseline feature-processing configuration uses:

- Original Recency values
- log1p transformation of Frequency
- log1p transformation of Monetary
- StandardScaler normalization

These transformations reduce the influence of highly skewed purchase distributions and place the features on comparable numerical scales.

Clustering is performed using the three transformed RFM features. Principal Component Analysis (PCA) is used for two-dimensional visualization, not as a replacement for the clustering features.

## 5. Clustering Algorithms

### 5.1 K-Means Clustering

K-Means partitions customers into a predefined number of clusters by minimizing within-cluster variation.

The implementation evaluates candidate values of k and compares clustering solutions using quality metrics and customer segment characteristics.

**Strengths:** Computational efficiency, straightforward interpretation, and compact cluster assignments.

**Limitations:** Requires specifying the number of clusters and may be sensitive to outliers and cluster geometry.

### 5.2 DBSCAN

DBSCAN identifies dense groups of customers based on neighborhood relationships.

Its principal parameters are:

- eps: Neighborhood radius
- min_samples: Minimum number of neighboring samples needed for core-point density

DBSCAN can identify irregularly shaped clusters and label some customers as noise.

**Strengths:** Does not require specifying the number of clusters and explicitly identifies noise.

**Limitations:** Sensitive to feature scaling and density parameters; some customers may remain unassigned.

### 5.3 Agglomerative Hierarchical Clustering

Agglomerative clustering progressively combines observations into larger groups based on their similarity.

The experiments examine cluster counts and linkage strategies, including Ward, complete, average, and single linkage.

**Strengths:** Supports multiple linkage definitions and provides a hierarchical perspective on customer similarity.

**Limitations:** Can be computationally expensive and sensitive to the selected linkage criterion.

## 6. Model Evaluation

The project uses several complementary evaluation measures.

### Silhouette Score

Measures how closely an observation matches its own cluster relative to other clusters.

Higher values generally indicate better separation.

### Davies–Bouldin Index

Measures cluster similarity based on within-cluster dispersion and between-cluster separation.

Lower values are preferred.

### Calinski–Harabasz Index

Compares between-cluster dispersion with within-cluster dispersion.

Higher values generally indicate more distinct clustering structures.

### Additional Evaluation Criteria

- Number of identified clusters
- Customer coverage
- Number of noise observations
- Model fitting time
- Customer segment sizes
- RFM-based cluster profiles

For DBSCAN, internal quality metrics are computed on assigned customers, excluding noise. Coverage is reported separately to avoid interpreting a high quality score without considering excluded customers.

Metrics that require at least two valid clusters are treated as undefined when that requirement is not met.

## 7. Cluster Stability Analysis

A clustering solution should ideally remain reasonably consistent when the dataset changes slightly.

To evaluate this property, the project performs **12 seeded subsampling trials**, each using **80% of customers**.

The Adjusted Rand Index (ARI) measures agreement between cluster assignments while accounting for agreement expected by chance.

Assignments are compared using matching customer identifiers, rather than assuming that cluster numbers have the same meaning across model fits.

The analysis records stability statistics and related coverage information.

For DBSCAN, stability measurements based only on assigned customers must be interpreted alongside noise and coverage diagnostics.

## 8. Original Technical Contribution: QSCS

### Quality–Stability–Coverage Score

A central contribution of this project is the proposed **Quality–Stability–Coverage Score (QSCS)**.

Standard clustering evaluation often emphasizes internal cluster quality. However, a clustering solution may achieve strong separation while producing unstable assignments or excluding a substantial proportion of customers.

QSCS combines three components:

- **Q — Quality:** A normalized combination of Silhouette, inverse Davies–Bouldin, and Calinski–Harabasz metrics.
- **S — Stability:** A normalized measure derived from Adjusted Rand Index.
- **C — Coverage:** The proportion of customers assigned to non-noise clusters.

### Proposed Formula

QSCS = 0.45 × Q + 0.35 × S + 0.20 × C

The weights assign 45% importance to clustering quality, 35% to stability, and 20% to customer coverage.

The project also examines alternative weighting configurations to determine how sensitive algorithm rankings are to the chosen priorities.

QSCS is a **project-specific experimental evaluation method**, not an externally validated universal metric. Its normalization depends on the candidate models being compared, and its weights represent explicit research choices.

## 9. Preprocessing Sensitivity Experiments

Customer segmentation results can change substantially depending on how features are transformed.

The project evaluates five preprocessing configurations:

1. Baseline transformation
2. Log transformations applied to all RFM features
3. Raw feature scaling
4. Robust scaling
5. Winsorization-based preprocessing

Each configuration is evaluated using the selected K-Means, DBSCAN, and Hierarchical Clustering models.

This produces **15 algorithm-by-preprocessing comparisons**.

The experiments measure changes in clustering quality, customer coverage, execution time, and assignment agreement with the baseline using ARI.

Selected model parameters are held constant across transformations to isolate preprocessing effects. These experiments therefore measure sensitivity under fixed parameters rather than the best result obtainable after retuning each variant.

## 10. Project Structure

The repository is organized as follows:

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
│   └── test_dashboard_logic.py
├── notebooks/
│   └── 01_customer_segmentation.ipynb
├── data/
│   └── raw/
└── results/
    ├── tables/
    └── final/
        ├── algorithm_comparison.csv
        ├── parameter_search.csv
        ├── stability_summary.csv
        ├── stability_trials.csv
        ├── qscs_comparison.csv
        ├── qscs_weight_sensitivity.csv
        ├── feature_sensitivity.csv
        ├── kmeans_profiles.csv
        ├── dbscan_profiles.csv
        ├── hierarchical_profiles.csv
        └── figures/
```

### Important Files

| File | Purpose |
| --- | --- |
| main.py | Original preprocessing, RFM, and K-Means analysis |
| final_main.py | Full comparative clustering experiments |
| part2_stability_qscs.py | Stability evaluation and QSCS computation |
| part3_sensitivity.py | Preprocessing sensitivity analysis |
| part4_report.py | Generation of experimental figures and QSCS weight comparisons |
| app.py | Interactive Streamlit application |
| dashboard_logic.py | Testable analytical operations used by Streamlit |
| demo.py | Console-based summary of saved experimental results |
| src/ | Core data processing, modeling, and evaluation implementation |
| tests/ | Automated tests |
| results/final/ | Experimental data, comparisons, profiles, and visualizations |

## 11. Software Requirements

The project uses Python 3.11 or later, with Python 3.12 recommended.

Primary libraries include:

- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn
- openpyxl
- Streamlit
- Plotly
- pytest
- Jupyter

Package requirements are maintained in `requirements.txt`.

The existing `requirements-lock.txt` reflects an earlier environment snapshot and does not include the later Streamlit additions. Use `requirements.txt` for the complete application.

## 12. Installation

Clone the repository:

```bash
git clone https://github.com/satnamkhokhar/customer-segmentation-datamining.git
cd customer-segmentation-datamining
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows, activate the environment using the appropriate command for Command Prompt or PowerShell.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

All commands in this README should be executed from the **repository root directory**.

### Dataset Setup

The original Excel workbook is not included in the Git repository.

Download the Online Retail dataset from the UCI Machine Learning Repository, extract the Excel workbook, and place it at:

```text
data/raw/Online Retail.xlsx
```

Alternatively, download it using:

```bash
python main.py --download
```

An internet connection is required for initial dataset download and dependency installation. Once the dataset and dependencies are available, the analyses can run locally.

## 13. Running the Project

### 13.1 Interactive Streamlit Application

Launch the application:

```bash
python -m streamlit run app.py
```

The application is typically available at:

http://localhost:8501

The interface contains six analytical sections:

| Section | Functionality |
| --- | --- |
| Overview & RFM | Data cleaning statistics and RFM distributions |
| Live Clustering | Fit clustering models using selectable algorithms, parameters, and preprocessing |
| Algorithm Comparison | Compare saved full-dataset clustering metrics and segment profiles |
| Stability & QSCS | Explore stability statistics and adjust composite score weights |
| Preprocessing Sensitivity | Examine changes in clustering results across feature transformations |
| Code & Methodology | Review reproducibility instructions and methodological considerations |

The Live Clustering section performs actual model fitting on the selected customer sample.

The comparison sections use saved experimental results so that the application can display full-dataset analyses without rerunning the complete parameter search during every interaction.

### 13.2 Run the Original Pipeline

```bash
python main.py
```

### 13.3 Run the Full Comparative Analysis

```bash
python final_main.py --repeats 12
```

This executes the comparative clustering research workflow, including model selection and stability evaluation.

The full experiment can be computationally expensive.

### 13.4 Run Stability and QSCS Experiments

```bash
python part2_stability_qscs.py --repeats 12
```

This uses the saved parameter search to evaluate selected clustering configurations.

### 13.5 Run Preprocessing Sensitivity Analysis

```bash
python part3_sensitivity.py
```

### 13.6 Regenerate Figures

```bash
python part4_report.py
```

This regenerates comparison figures and the QSCS weight-sensitivity output using saved experimental tables.

### 13.7 Inspect Saved Results

```bash
python demo.py
```

This displays saved clustering comparisons, stability summaries, QSCS scores, and customer segment profiles without repeating expensive model fitting.

**Note:** Rerunning experiment scripts may overwrite existing result files. Preserve copies of results when comparing different experimental configurations.

## 14. Automated Testing

The project includes automated tests for:

- Data preprocessing and validation
- Clustering and research calculations
- Stability analysis
- QSCS computation
- Preprocessing sensitivity
- Dashboard analytical functions

Run the complete test suite:

```bash
python -m pytest -q
```

The current integrated project was locally tested with **33 passing tests**.

The tests provide implementation-level validation; they do not establish that any particular segmentation is objectively correct or commercially optimal.

## 15. Experimental Outputs

The `results/final/` directory contains the saved results of the comparative research.

Important outputs include:

- `algorithm_comparison.csv`: Selected algorithm performance metrics
- `parameter_search.csv`: Evaluated clustering configurations
- `stability_trials.csv`: Individual stability experiments
- `stability_summary.csv`: Aggregated stability statistics
- `qscs_comparison.csv`: Quality, stability, coverage, and QSCS results
- `qscs_weight_sensitivity.csv`: Alternative QSCS weighting scenarios
- `feature_sensitivity.csv`: Preprocessing sensitivity comparisons
- `kmeans_profiles.csv`: K-Means segment characteristics
- `dbscan_profiles.csv`: DBSCAN segment characteristics
- `hierarchical_profiles.csv`: Hierarchical segment characteristics
- `figures/`: Saved visual comparisons

The original mid-project K-Means experiment selected k = 3, while the final comparative search selected k = 2 using a different model-selection procedure.

These represent different stages of the analysis and should not be interpreted as identical experiments.

## 16. Research Limitations

Several limitations should be considered when interpreting the findings.

1. **No external ground-truth labels:** The dataset does not provide objectively correct customer segment labels.
2. **Dependence on preprocessing:** Scaling, transformations, and outlier treatment can change clustering outcomes.
3. **Internal metric limitations:** Strong clustering scores do not automatically establish business usefulness.
4. **DBSCAN coverage:** Density-based clustering may leave some customers classified as noise.
5. **Stability interpretation:** Assignment agreement under subsampling is informative but does not guarantee stability under future customer behavior.
6. **QSCS weighting:** Composite scores depend on researcher-selected weights and normalization boundaries.
7. **Historical dataset:** The results describe purchasing behavior in the dataset's original time period, not current consumer behavior.
8. **Business interpretation:** The project identifies purchasing-behavior groups but does not establish causal relationships or predict churn.

## 17. Conclusion

This project provides a reproducible comparison of K-Means, DBSCAN, and Agglomerative Hierarchical Clustering for customer segmentation using real retail purchasing data.

Its principal contribution is an evaluation framework that considers not only cluster separation but also assignment stability and customer coverage.

By combining conventional clustering metrics, repeated stability experiments, preprocessing sensitivity analysis, and the proposed QSCS method, the project supports a more comprehensive comparison of unsupervised customer segmentation techniques.

The implementation includes reusable analytical code, automated tests, saved experiment results, and an interactive Streamlit application for exploring the research.


## Submission Package (Sources)

The submission ZIP contains a `Sources/` directory with the runnable implementation, `src/`, `tests/`, saved experimental results, README, and the original UCI Online Retail workbook. Open a terminal inside `Sources/` and follow Sections 12–14 to install dependencies, run tests, and launch the application. Python 3.12 is recommended. The workbook is included in the submission ZIP but excluded from the public Git repository.

The final selected-model quality metrics have been regenerated and checked against the baseline rows in the preprocessing sensitivity analysis.

## 18. References

Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

scikit-learn Developers. *scikit-learn User Guide*. https://scikit-learn.org/stable/user_guide.html

Streamlit. *Streamlit Documentation*. https://docs.streamlit.io/
