"""Final research experiments. No synthetic values are used for research outputs."""
from __future__ import annotations
import json, platform, hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .rfm_analysis import FEATURES

SEED=42

def features_for(rfm, variant='baseline'):
    x=rfm[FEATURES].astype(float).copy()
    if variant in ('baseline','robust','winsor'):
        x[['Frequency','Monetary']]=np.log1p(x[['Frequency','Monetary']])
    elif variant=='all_log':
        x=np.log1p(x)
    elif variant!='raw':
        raise ValueError('Unknown variant '+variant)
    if variant=='winsor':
        x=x.clip(lower=x.quantile(.01),upper=x.quantile(.99),axis=1)
    scaler=RobustScaler() if variant=='robust' else StandardScaler()
    return scaler.fit_transform(x)

def metrics(x, labels, silhouette_sample=300):
    """Compute assigned-customer quality, with a cluster-aware silhouette sample."""
    labels=np.asarray(labels)
    keep=labels!=-1
    assigned=labels[keep]
    count=len(set(assigned))
    out={'clusters':count,'coverage':float(keep.mean()),'noise_count':int((~keep).sum()),'evaluated_customers':int(keep.sum()),'silhouette':np.nan,'davies_bouldin':np.nan,'calinski_harabasz':np.nan}
    if count>=2 and len(assigned)>count:
        xx=x[keep]
        # Random samples can omit small clusters or include singletons; select
        # at least two observations per cluster when possible.
        rng=np.random.default_rng(SEED)
        chosen=[]
        for label in np.unique(assigned):
            idx=np.flatnonzero(assigned==label)
            chosen.extend(rng.choice(idx,size=min(2,len(idx)),replace=False).tolist())
        target=min(len(assigned),max(silhouette_sample,len(chosen)))
        remaining=np.setdiff1d(np.arange(len(assigned)),np.array(chosen,dtype=int))
        if target>len(chosen):
            chosen.extend(rng.choice(remaining,size=target-len(chosen),replace=False).tolist())
        sample=np.array(chosen,dtype=int)
        sample_labels=assigned[sample]
        if len(np.unique(sample_labels))>=2 and len(sample)>len(np.unique(sample_labels)):
            out['silhouette']=float(silhouette_score(xx[sample],sample_labels))
        out.update(davies_bouldin=float(davies_bouldin_score(xx,assigned)),calinski_harabasz=float(calinski_harabasz_score(xx,assigned)))
    return out

def fit_algorithm(x,algorithm,params):
    if algorithm=='kmeans':
        model=KMeans(n_clusters=int(params['k']),random_state=int(params.get('seed',SEED)),n_init=20,algorithm='lloyd')
    elif algorithm=='dbscan':
        model=DBSCAN(eps=float(params['eps']),min_samples=int(params['min_samples']))
    elif algorithm=='hierarchical':
        model=AgglomerativeClustering(n_clusters=int(params['k']),linkage=params['linkage'])
    else: raise ValueError(algorithm)
    t=perf_counter();labels=model.fit_predict(x);elapsed=perf_counter()-t
    return labels,elapsed

def profile(rfm,labels):
    d=rfm.copy();d['Cluster']=labels
    z=d.groupby('Cluster').agg(customers=('CustomerID','size'),mean_recency=('Recency','mean'),median_recency=('Recency','median'),mean_frequency=('Frequency','mean'),median_frequency=('Frequency','median'),mean_monetary=('Monetary','mean'),median_monetary=('Monetary','median'),total_monetary=('Monetary','sum')).reset_index()
    z['percentage']=z.customers/len(d)*100
    return z

def evaluate_grid(x):
    rows=[]; labels_by_key={}; configs={}
    # all algorithms on the same 3-dimensional customer representation
    configs['kmeans']=[{'k':k} for k in range(2,7)]
    configs['hierarchical']=[{'k':k,'linkage':link} for link in ('ward','complete','average') for k in (2,3,4)]
    distances=NearestNeighbors(n_neighbors=15).fit(x).kneighbors(x)[0]
    eps_values=np.unique(np.round(np.quantile(distances[:,7],[.25,.4,.55,.7,.8,.9,.95,.98, .995]),4))
    configs['dbscan']=[{'eps':float(eps),'min_samples':m} for m in (4,8) for eps in eps_values if eps>0]
    for algorithm,grid in configs.items():
        print('GRID',algorithm,len(grid),flush=True)
        for i,p in enumerate(grid):
            print('CONFIG',algorithm,i,flush=True)
            key=f'{algorithm}_{i:03d}'
            try:
                labels,fit=fit_algorithm(x,algorithm,p)
                t=perf_counter();m=metrics(x,labels);evaltime=perf_counter()-t
                rows.append({'key':key,'algorithm':algorithm,**p,**m,'fit_seconds':fit,'evaluation_seconds':evaltime,'valid':bool(np.isfinite(m['silhouette'])),'reason':'' if np.isfinite(m['silhouette']) else 'fewer than 2 usable clusters'})
                labels_by_key[key]=labels
            except Exception as e:
                rows.append({'key':key,'algorithm':algorithm,**p,'valid':False,'reason':str(e)})
    return pd.DataFrame(rows),labels_by_key,eps_values

def select_models(grid):
    chosen={}
    for alg in ('kmeans','dbscan','hierarchical'):
        d=grid[(grid.algorithm==alg)&(grid.valid==True)].copy()
        if alg=='dbscan':
            # coverage floor prevents selecting tiny easy subsets
            d=d[(d.coverage>=.80)&(d.clusters>=2)]
        if len(d)==0:
            chosen[alg]=None;continue
        d['rank_score']=d.silhouette - .1*(1-d.coverage)
        if alg=='hierarchical':
            # break near ties in favor of Ward linkage and fewer clusters
            d['ward_bonus']=(d.linkage=='ward')*.00001
            d['rank_score']+=d.ward_bonus
        row=d.sort_values(['rank_score','coverage','key'],ascending=[False,False,True]).iloc[0]
        chosen[alg]=row['key']
    return chosen

def stability(x,rfm,selected,grid,repeats=12,frac=.8):
    ids=rfm.CustomerID.astype(str).to_numpy()
    if len(set(ids))!=len(ids): raise ValueError('Duplicate customer identifiers')
    records=[]
    for alg,key in selected.items():
        if key is None: continue
        row=grid.set_index('key').loc[key]
        params={'k':int(row.k)} if alg=='kmeans' else ({'eps':float(row.eps),'min_samples':int(row.min_samples)} if alg=='dbscan' else {'k':int(row.k),'linkage':row.linkage})
        reference,_=fit_algorithm(x,alg,params)
        for rep in range(repeats):
            rng=np.random.default_rng(SEED+rep)
            ix=np.sort(rng.choice(len(ids),size=max(3,int(len(ids)*frac)),replace=False))
            # Fit to a subset, then compare labels for exactly these customer IDs
            subparams={**params,'seed':SEED+rep} if alg=='kmeans' else params
            trial,_=fit_algorithm(x[ix],alg,subparams)
            ref=reference[ix]
            both=(ref!=-1)&(trial!=-1)
            common=float(both.mean())
            ari=float(adjusted_rand_score(ref[both],trial[both])) if both.sum()>2 and len(set(ref[both]))>1 and len(set(trial[both]))>1 else np.nan
            full=float(adjusted_rand_score(ref,trial)) if len(ref)>2 else np.nan
            noise_agreement=float(np.mean((ref==-1)==(trial==-1)))
            records.append({'algorithm':alg,'repeat':rep,'sample_n':len(ix),'common_assigned_n':int(both.sum()),'common_assigned_fraction':common,'ari_assigned':ari,'ari_all_including_noise':full,'noise_status_agreement':noise_agreement})
    trials=pd.DataFrame(records)
    summary=trials.groupby('algorithm').agg(ari_mean=('ari_assigned','mean'),ari_std=('ari_assigned','std'),ari_median=('ari_assigned','median'),ari_min=('ari_assigned','min'),ari_max=('ari_assigned','max'),valid_trials=('ari_assigned','count'),coverage_overlap=('common_assigned_fraction','mean'),noise_agreement=('noise_status_agreement','mean')).reset_index()
    return trials,summary

def qscs(grid,selected,stability_summary,weights=(.45,.35,.20)):
    """Candidate-pool min/max quality; explicit normalization and undefined handling."""
    if any(w<0 for w in weights) or not np.isclose(sum(weights),1):raise ValueError('Weights must be nonnegative and sum to 1')
    valid=grid[grid.valid==True]
    bounds={}
    for col in ('silhouette','davies_bouldin','calinski_harabasz'):
        a=valid[col].dropna();bounds[col]={'min':float(a.min()),'max':float(a.max())}
    rows=[]
    for alg,key in selected.items():
        if key is None:continue
        r=grid.set_index('key').loc[key]
        scores=[]
        for col in bounds:
            lo,hi=bounds[col]['min'],bounds[col]['max']
            val=.5 if hi==lo else (float(r[col])-lo)/(hi-lo)
            scores.append(1-val if col=='davies_bouldin' else val)
        Q=float(np.mean(scores));stab=stability_summary.set_index('algorithm').loc[alg,'ari_mean']
        S=float((stab+1)/2) if np.isfinite(stab) else np.nan
        C=float(r.coverage)
        rows.append({'algorithm':alg,'key':key,'Q':Q,'S':S,'C':C,'QSCS':float(weights[0]*Q+weights[1]*S+weights[2]*C) if np.isfinite(S) else np.nan})
    return pd.DataFrame(rows),bounds

def save_figure(path, draw):
    fig,ax=plt.subplots(figsize=(9,5));draw(ax);fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)

def run_final(rfm,out,repeats=12,fast=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    x=features_for(rfm)
    grid,labels,eps=evaluate_grid(x)
    grid.to_csv(out/'parameter_search.csv',index=False)
    selected=select_models(grid)
    profiles={}
    for alg,key in selected.items():
        if key is None:continue
        profiles[alg]=profile(rfm,labels[key]);profiles[alg].to_csv(out/f'{alg}_profiles.csv',index=False)
        pd.DataFrame({'CustomerID':rfm.CustomerID,'Cluster':labels[key]}).to_csv(out/f'{alg}_assignments.csv',index=False)
    trials,stab=stability(x,rfm,selected,grid,repeats=repeats)
    trials.to_csv(out/'stability_trials.csv',index=False);stab.to_csv(out/'stability_summary.csv',index=False)
    scores,bounds=qscs(grid,selected,stab);scores.to_csv(out/'qscs_comparison.csv',index=False)
    variants=[]
    for variant in ('baseline','all_log','raw','robust','winsor'):
        xx=features_for(rfm,variant)
        for alg,key in selected.items():
            if key is None:continue
            row=grid.set_index('key').loc[key]
            p={'k':int(row.k)} if alg=='kmeans' else ({'eps':float(row.eps),'min_samples':int(row.min_samples)} if alg=='dbscan' else {'k':int(row.k),'linkage':row.linkage})
            lab,t=fit_algorithm(xx,alg,p)
            m=metrics(xx,lab)
            variants.append({'variant':variant,'algorithm':alg,**m,'ari_vs_baseline':float(adjusted_rand_score(labels[key],lab)),'fit_seconds':t})
    pd.DataFrame(variants).to_csv(out/'feature_sensitivity.csv',index=False)
    weights=[(.45,.35,.20),(.333333333333,.333333333334,.333333333333),(.7,.2,.1),(.2,.7,.1),(.2,.2,.6),(.65,.35,0),(.65,0,.35),(0,.65,.35)]
    ablations=[]
    for w in weights:
        z=scores.copy();z['score']=w[0]*z.Q+w[1]*z.S+w[2]*z.C
        for _,r in z.iterrows():ablations.append({'weights':str(w),'algorithm':r.algorithm,'score':r.score})
    pd.DataFrame(ablations).to_csv(out/'qscs_weight_sensitivity.csv',index=False)
    selectedrows=grid[grid.key.isin([v for v in selected.values() if v])].copy()
    selectedrows.to_csv(out/'algorithm_comparison.csv',index=False)
    save_figure(figdir/'algorithm_silhouette.png',lambda ax:(ax.bar(selectedrows.algorithm,selectedrows.silhouette),ax.set(ylabel='Silhouette (assigned customers)',title='Selected algorithm quality')))
    save_figure(figdir/'stability.png',lambda ax:(ax.bar(stab.algorithm,stab.ari_mean),ax.set(ylabel='Mean ARI on common assigned customers',title='Subsample stability')))
    save_figure(figdir/'qscs.png',lambda ax:(ax.bar(scores.algorithm,scores.QSCS),ax.set(ylabel='QSCS',title='Proposed quality-stability-coverage score')))
    save_figure(figdir/'dbscan_noise.png',lambda ax:(ax.scatter(grid.loc[grid.algorithm=='dbscan','eps'],1-grid.loc[grid.algorithm=='dbscan','coverage'],s=15),ax.set(xlabel='eps',ylabel='Noise fraction',title='DBSCAN noise sensitivity')))
    # Dendrogram on a fixed subsample; labels never used to fit a model in the figure.
    rng=np.random.default_rng(SEED);ix=np.sort(rng.choice(len(x),size=min(350,len(x)),replace=False))
    Z=linkage(x[ix],method='ward')
    fig,ax=plt.subplots(figsize=(11,5));dendrogram(Z,truncate_mode='lastp',p=25,ax=ax);ax.set(title=f'Truncated Ward dendrogram ({len(ix)} sampled customers)',ylabel='Merge distance');fig.tight_layout();fig.savefig(figdir/'dendrogram.png',dpi=200);plt.close(fig)
    manifest={'selected':selected,'qscs_weights':{'quality':.45,'stability':.35,'coverage':.20},'quality_normalization_bounds':bounds,'seed':SEED,'stability_repeats':repeats,'subsample_fraction':.8,'customer_count':len(rfm),'grid_silhouette_sample_size':300,'python':platform.python_version(),'platform':platform.platform(),'limitations':['DBSCAN quality excludes noise; coverage reported separately','ARI assigned subset can differ by algorithm','QSCS normalization depends on candidate pool','No external business outcome labels']}
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'grid':grid,'selected':selected,'scores':scores,'stability':stab,'comparison':selectedrows,'manifest':manifest}
