import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
from src.final_research import features_for,metrics,fit_algorithm,qscs,stability

def fixture():
    rng=np.random.default_rng(9)
    n=45
    return pd.DataFrame({'CustomerID':[str(i) for i in range(n)],'Recency':np.r_[rng.integers(1,25,15),rng.integers(50,90,15),rng.integers(170,300,15)],'Frequency':np.r_[rng.integers(6,12,15),rng.integers(2,5,15),rng.integers(1,3,15)],'Monetary':np.r_[rng.integers(1000,4000,15),rng.integers(300,800,15),rng.integers(20,200,15)]})
def test_features():
    r=fixture()
    for variant in ('baseline','raw','all_log','robust','winsor'):
        x=features_for(r,variant)
        assert x.shape==(45,3) and np.isfinite(x).all()
def test_noise():
    x=features_for(fixture())
    assert metrics(x,np.full(45,-1))['clusters']==0
    assert np.isnan(metrics(x,np.full(45,-1))['silhouette'])
    assert metrics(x,np.r_[np.zeros(20),np.ones(20),np.full(5,-1)])['coverage']==40/45
def test_fit():
    x=features_for(fixture())
    for a,p in [('kmeans',{'k':3}),('hierarchical',{'k':3,'linkage':'ward'}),('dbscan',{'eps':.8,'min_samples':3})]:
        y,t=fit_algorithm(x,a,p)
        assert len(y)==45 and t>=0
def test_ari_permutation():
    assert adjusted_rand_score([0,0,1,1],[8,8,4,4])==1
def test_stability_ids():
    r=fixture();r.loc[1,'CustomerID']=r.loc[0,'CustomerID']
    with pytest.raises(ValueError,match='Duplicate'):
        stability(features_for(r),r,{},pd.DataFrame())
def test_score_weights():
    with pytest.raises(ValueError):qscs(pd.DataFrame(),{},pd.DataFrame(),weights=(1,1,1))
