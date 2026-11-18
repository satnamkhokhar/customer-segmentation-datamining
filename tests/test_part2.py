"""Focused tests for the Part 2 research contribution."""
import numpy as np
import pandas as pd
import pytest
from src.final_research import features_for, stability, qscs


def sample_rfm():
    rng=np.random.default_rng(123)
    n=90
    return pd.DataFrame({
        'CustomerID':[f'c{i:03}' for i in range(n)],
        'Recency':np.r_[rng.integers(1,25,30),rng.integers(80,140,30),rng.integers(200,320,30)],
        'Frequency':np.r_[rng.integers(8,15,30),rng.integers(3,7,30),rng.integers(1,3,30)],
        'Monetary':np.r_[rng.integers(2000,4500,30),rng.integers(300,900,30),rng.integers(20,200,30)],
    })


def test_stability_is_repeatable_and_tracks_matching_ids():
    r=sample_rfm();x=features_for(r)
    grid=pd.DataFrame([{'key':'kmeans_000','algorithm':'kmeans','k':3}])
    selected={'kmeans':'kmeans_000'}
    a,sa=stability(x,r,selected,grid,repeats=4)
    b,sb=stability(x,r,selected,grid,repeats=4)
    pd.testing.assert_frame_equal(a,b)
    pd.testing.assert_frame_equal(sa,sb)
    assert len(a)==4 and (a.sample_n==72).all()
    assert (a.common_assigned_n==72).all()
    assert (a.ari_assigned.between(-1,1)).all()
    assert (a.noise_status_agreement==1).all()


def test_qscs_known_values_and_missing_stability():
    grid=pd.DataFrame([
        {'key':'a','algorithm':'kmeans','valid':True,'silhouette':.8,'davies_bouldin':.2,'calinski_harabasz':90.,'coverage':1.},
        {'key':'b','algorithm':'dbscan','valid':True,'silhouette':.2,'davies_bouldin':1.2,'calinski_harabasz':10.,'coverage':.5},
    ])
    stab=pd.DataFrame([{'algorithm':'kmeans','ari_mean':1.},{'algorithm':'dbscan','ari_mean':0.}])
    scores,_=qscs(grid,{'kmeans':'a','dbscan':'b'},stab)
    a=scores.set_index('algorithm').loc['kmeans'];b=scores.set_index('algorithm').loc['dbscan']
    assert a.Q==pytest.approx(1) and a.S==pytest.approx(1) and a.QSCS==pytest.approx(1)
    assert b.Q==pytest.approx(0) and b.S==pytest.approx(.5) and b.QSCS==pytest.approx(.275)
    stab.loc[1,'ari_mean']=np.nan
    scores,_=qscs(grid,{'dbscan':'b'},stab)
    assert np.isnan(scores.iloc[0].QSCS)


def test_qscs_rejects_invalid_weights():
    with pytest.raises(ValueError):qscs(pd.DataFrame(),{},pd.DataFrame(),weights=(-.1,.6,.5))
