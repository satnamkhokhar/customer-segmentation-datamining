"""Execute the first progress-report experiment on real UCI transactions."""
import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from src.data_loader import load_data, download_dataset, explore
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm, FEATURES
from src.clustering import prepare_features, run_kmeans, profile_segments, select_k
from src.visualization import eda_figures, rfm_figures, clustering_figures

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    """Write portable JSON; reject NaN or infinite results."""
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v), allow_nan=False) + "\n")


def run_pipeline(dataset=None, download=False, selected_k=None, output_root=None):
    """Run shared CLI/notebook workflow; generated data never replaces the raw file."""
    output = Path(output_root) if output_root else ROOT
    dataset = Path(dataset) if dataset else ROOT / 'data/raw/Online Retail.xlsx'
    tables, figures, models, processed = [output / p for p in ['results/tables', 'results/figures', 'results/models', 'data/processed']]
    for path in [tables, figures, models, processed]:
        path.mkdir(parents=True, exist_ok=True)
    if download:
        download_dataset(dataset)
    raw = load_data(dataset)
    exploration = explore(raw)
    write_json(tables / 'exploration.json', exploration)
    eda_figures(raw, figures)
    clean, ledger = clean_transactions(raw)
    ledger.to_csv(tables / 'cleaning_summary.csv', index=False)
    clean.to_csv(processed / 'cleaned_transactions.csv', index=False)
    rfm, reference = calculate_rfm(clean)
    rfm.to_csv(processed / 'customer_rfm.csv', index=False)
    stats = rfm[FEATURES].describe().T
    stats['median'] = rfm[FEATURES].median()
    stats.to_csv(tables / 'rfm_statistics.csv')
    rfm_figures(rfm, figures)
    scaled, scaler, transformed = prepare_features(rfm)
    transformed.describe().T.to_csv(tables / 'transformed_statistics.csv')
    # Limit BLAS/OpenMP oversubscription for this small, three-feature experiment.
    with threadpool_limits(limits=2):
        comparison, fitted, assignments = run_kmeans(scaled)
    comparison.to_csv(tables / 'kmeans_comparison.csv', index=False)
    profiles = {}
    for k, labels in assignments.items():
        assigned, profile = profile_segments(rfm, labels)
        assigned.to_csv(processed / f'customer_assignments_k{k}.csv', index=False)
        profile.to_csv(tables / f'segment_profiles_k{k}.csv', index=False)
        profiles[k] = profile
    k, rationale, elbow = select_k(comparison, profiles)
    if selected_k is not None:
        if selected_k not in fitted:
            raise ValueError('--k must be one of the tested K values')
        k = selected_k
        rationale = f'Explicit researcher override K={k}. Review saved comparison and profiles before reporting. Automatic rule: ' + rationale
    assigned, profile = profile_segments(rfm, assignments[k])
    assigned.to_csv(processed / 'customer_segments.csv', index=False)
    profile.to_csv(tables / 'segment_profiles.csv', index=False)
    with (models / 'selected_kmeans.pkl').open('wb') as handle:
        pickle.dump(fitted[k], handle)
    with (models / 'scaler.pkl').open('wb') as handle:
        pickle.dump(scaler, handle)
    configuration = {'features': FEATURES, 'log1p': ['Frequency', 'Monetary'], 'recency': 'unchanged calendar days',
                     'outlier_treatment': 'none; all valid customers retained', 'random_state': 42, 'n_init': 20,
                     'scaler': 'StandardScaler', 'threads': 2, 'reference_date': str(reference.date()),
                     'recency_skewness': float(rfm.Recency.skew()),
                     'log1p_recency_skewness': float(np.log1p(rfm.Recency).skew()),
                     'recency_decision': 'Retained untransformed for an interpretable baseline; sensitivity analysis deferred.',
                     'raw_skewness': rfm[FEATURES].skew().to_dict(),
                     'transformed_skewness': transformed.skew().to_dict(),
                     'iqr_outlier_counts': {c: int(((rfm[c] < rfm[c].quantile(.25)-1.5*(rfm[c].quantile(.75)-rfm[c].quantile(.25))) | (rfm[c] > rfm[c].quantile(.75)+1.5*(rfm[c].quantile(.75)-rfm[c].quantile(.25)))).sum()) for c in FEATURES}}
    write_json(models / 'transformation.json', configuration)
    variance = clustering_figures(scaled, assignments[k], comparison, profile, figures)
    chosen = comparison.loc[comparison.k.eq(k)].iloc[0]
    metrics = {key: float(chosen[key]) for key in ['inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'fit_seconds']}
    summary = {'dataset': {'original_transactions': len(raw), 'cleaned_transactions': len(clean),
                           'removed_records': len(raw)-len(clean), 'customers': len(rfm),
                           'sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                           'source': 'https://archive.ics.uci.edu/dataset/352/online+retail'},
               'rfm': {'reference_date': str(reference.date()), 'statistics': stats.to_dict(), 'clustering_customers': len(rfm)},
               'kmeans': {'tested_k': comparison.k.tolist(), 'selected_k': k, 'rationale': rationale,
                          'elbow_k': elbow, **metrics, 'profiles': profile.to_dict(orient='records')},
               'pca_visualization_explained_variance': variance,
               'versions': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','matplotlib','seaborn','openpyxl']}},
               'timing_scope': 'KMeans fit_predict only; excludes metrics, loading and plotting',
               'limitations': ['Returns excluded; Monetary is valid positive purchase value, not net revenue.',
                              'No demonstrated churn or cluster stability claim.', 'No automatic outlier removal.',
                              'K selection is provisional and requires domain review.']}
    write_json(tables / 'experimental_summary.json', summary)
    profile_text = profile.to_string(index=False, float_format=lambda x: f'{x:.2f}')
    report = f'''# First progress report: customer segmentation

## 1. Project's Current Update
Executed real UCI Online Retail acquisition, exploration, sequential cleaning, RFM engineering, feature transformation and scaling, K-Means K={comparison.k.tolist()}, evaluation, profiling, and figure/table exports.
Dataset SHA-256: `{summary['dataset']['sha256']}`. Reference date: **{reference.date()}** (one day after the latest valid purchase, using calendar-day recency).

## 2. Results Obtained
Original transaction rows: **{len(raw):,}**. Cleaned rows: **{len(clean):,}**. Removed: **{len(raw)-len(clean):,}**. Customers clustered: **{len(rfm):,}**.

{rationale}

Selected-model metrics: {json.dumps(metrics)}. Fit time excludes evaluation, loading, and plotting.

```text
{profile_text}
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
'''
    (output / 'results/progress_summary.md').write_text(report)
    print(json.dumps({'transactions': len(raw), 'cleaned': len(clean), 'customers': len(rfm), 'selected_k': k, **metrics}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download official UCI dataset only if missing')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--k', type=int, help='Optional researcher override from tested K values')
    args = parser.parse_args()
    run_pipeline(args.dataset, args.download, args.k)
