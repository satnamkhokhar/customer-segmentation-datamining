"""Part 3: independently reproducible preprocessing sensitivity and comparisons.

Uses the Part 1 parameter search to avoid re-tuning parameters on each variant.
Refits the baseline and every variant to the same customers; ARI is label invariant.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from src.final_research import features_for, fit_algorithm, metrics, select_models
from src.data_loader import load_data
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm

VARIANTS=('baseline','all_log','raw','robust','winsor')

def selected_params(algorithm, row):
    if algorithm=='kmeans': return {'k':int(row['k'])}
    if algorithm=='hierarchical': return {'k':int(row['k']),'linkage':str(row['linkage'])}
    if algorithm=='dbscan': return {'eps':float(row['eps']),'min_samples':int(row['min_samples'])}
    raise ValueError(f'Unknown algorithm: {algorithm}')

def compare_sensitivity(rfm, grid, selected=None):
    if selected is None: selected=select_models(grid)
    baseline_x=features_for(rfm,'baseline')
    rows=[]; baselines={}
    for alg,key in selected.items():
        if key is None: continue
        matches=grid.loc[grid.key==key]
        if len(matches)!=1: raise ValueError(f'Missing or duplicate grid key {key}')
        params=selected_params(alg,matches.iloc[0])
        baselines[alg],_=fit_algorithm(baseline_x,alg,params)
        for variant in VARIANTS:
            xx=features_for(rfm,variant)
            lab,elapsed=fit_algorithm(xx,alg,params)
            quality=metrics(xx,lab)
            rows.append({'algorithm':alg,'key':key,'variant':variant,
                         'ari_vs_baseline':float(adjusted_rand_score(baselines[alg],lab)),
                         'fit_seconds':elapsed,**quality})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',default='data/raw/Online Retail.xlsx')
    ap.add_argument('--grid',default='results/final/parameter_search.csv')
    ap.add_argument('--out',default='results/final/feature_sensitivity.csv')
    args=ap.parse_args()
    raw=load_data(Path(args.dataset))
    cleaned,_=clean_transactions(raw)
    rfm,_=calculate_rfm(cleaned)
    grid=pd.read_csv(args.grid)
    comparison=compare_sensitivity(rfm,grid)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(dest,index=False)
    print(comparison[['algorithm','variant','clusters','coverage','silhouette','ari_vs_baseline']].to_string(index=False))
    print(f'Saved {len(comparison)} experiments to {dest}')

if __name__=='__main__': main()
