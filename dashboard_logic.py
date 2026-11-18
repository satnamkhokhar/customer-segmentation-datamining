"""Testable dashboard operations, independent of the Streamlit UI."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.final_research import features_for, fit_algorithm, metrics, profile

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'/'final'


def read_saved(name):
    allowed={'cleaning_summary.csv','rfm_statistics.csv','algorithm_comparison.csv',
             'parameter_search.csv','stability_summary.csv','stability_trials.csv',
             'qscs_comparison.csv','feature_sensitivity.csv','kmeans_profiles.csv',
             'dbscan_profiles.csv','hierarchical_profiles.csv'}
    if name not in allowed:
        raise ValueError('Not an approved results file')
    return pd.read_csv(RESULTS/name)


def run_live(rfm,algorithm,variant='baseline',k=3,eps=.6,min_samples=4,linkage='ward'):
    if len(rfm)<5:
        raise ValueError('At least five customers are required')
    x=features_for(rfm,variant)
    if algorithm=='kmeans': params={'k':int(k)}
    elif algorithm=='dbscan':params={'eps':float(eps),'min_samples':int(min_samples)}
    elif algorithm=='hierarchical':params={'k':int(k),'linkage':linkage}
    else:raise ValueError('Unknown algorithm')
    labels,seconds=fit_algorithm(x,algorithm,params)
    quality=metrics(x,labels)
    points=PCA(n_components=2,random_state=42).fit_transform(x)
    scatter=pd.DataFrame({'PC1':points[:,0],'PC2':points[:,1],
                          'Cluster':np.where(labels==-1,'Noise',labels.astype(str)),
                          'CustomerID':rfm.CustomerID.astype(str)})
    return quality,seconds,profile(rfm,labels),scatter


def weighted_scores(scores,quality=.45,stability=.35,coverage=.20):
    weights=np.array([quality,stability,coverage],dtype=float)
    if np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('Weights must be nonnegative and sum to 1')
    required={'algorithm','Q','S','C'}
    if not required.issubset(scores.columns):raise ValueError('Missing QSCS components')
    out=scores[['algorithm','Q','S','C']].copy()
    out['custom_score']=out.Q*quality+out.S*stability+out.C*coverage
    return out.sort_values('custom_score',ascending=False).reset_index(drop=True)
