"""Fast demonstration of completed clustering results (no expensive retraining)."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'final'
REQUIRED=['parameter_search.csv','algorithm_comparison.csv','stability_trials.csv','stability_summary.csv','qscs_comparison.csv','feature_sensitivity.csv','kmeans_profiles.csv','dbscan_profiles.csv','hierarchical_profiles.csv']

def main():
    missing=[x for x in REQUIRED if not (OUT/x).exists()]
    if missing: raise SystemExit('Missing experiment outputs: '+', '.join(missing))
    print('\nCSC 4740 | Customer Segmentation: Demonstration')
    print('Algorithms: K-Means, DBSCAN, Hierarchical Clustering')
    for name,cols in [('algorithm_comparison.csv',['algorithm','clusters','coverage','silhouette','davies_bouldin','calinski_harabasz']),('stability_summary.csv',['algorithm','ari_mean','valid_trials','coverage_overlap']),('qscs_comparison.csv',['algorithm','Q','S','C','QSCS'])]:
        df=pd.read_csv(OUT/name)
        print('\n'+name.replace('_',' ').replace('.csv','').title())
        print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
    df=pd.read_csv(OUT/'feature_sensitivity.csv')
    print('\nPreprocessing sensitivity: '+str(len(df))+' experiments')
    for alg in ('kmeans','dbscan','hierarchical'):
        df=pd.read_csv(OUT/f'{alg}_profiles.csv')
        print(f'\n{alg.upper()} segment profiles:')
        print(df[[c for c in ['Cluster','customers','median_recency','median_frequency','median_monetary'] if c in df.columns]].round(2).to_string(index=False))
    figs=sorted((OUT/'figures').glob('*.png'))
    print('\nSaved charts: '+str(len(figs)))
    for fig in figs:print('  '+str(fig.relative_to(ROOT)))
    print('\nDemo verified. To regenerate plots: python part4_report.py')
    print('To run tests: python -m pytest -q')
    print('To rerun the full research: python final_main.py --repeats 12')

if __name__=='__main__':main()
