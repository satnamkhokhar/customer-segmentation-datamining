import numpy as np
import pandas as pd
import pytest
from part3_sensitivity import compare_sensitivity, selected_params, VARIANTS

def rfm_fixture():
    rng=np.random.default_rng(18)
    return pd.DataFrame({'CustomerID':range(60),
        'Recency':np.r_[rng.integers(1,15,20),rng.integers(80,120,20),rng.integers(220,280,20)],
        'Frequency':np.r_[rng.integers(8,12,20),rng.integers(3,6,20),rng.integers(1,3,20)],
        'Monetary':np.r_[rng.integers(2000,3000,20),rng.integers(400,700,20),rng.integers(30,90,20)]})

def test_selected_parameters():
    assert selected_params('kmeans',{'k':3.0})=={'k':3}
    assert selected_params('dbscan',{'eps':.4,'min_samples':5.0})=={'eps':.4,'min_samples':5}
    assert selected_params('hierarchical',{'k':2.0,'linkage':'ward'})=={'k':2,'linkage':'ward'}
    with pytest.raises(ValueError): selected_params('unknown',{})

def test_sensitivity_uses_identical_customers_and_baseline_ari_one():
    grid=pd.DataFrame([{'key':'k1','algorithm':'kmeans','k':3}])
    df=compare_sensitivity(rfm_fixture(),grid,{'kmeans':'k1'})
    assert set(df.variant)==set(VARIANTS)
    assert len(df)==5
    assert df.loc[df.variant=='baseline','ari_vs_baseline'].iloc[0]==pytest.approx(1)
    assert (df.evaluated_customers==60).all()
    assert df.ari_vs_baseline.between(-1,1).all()

def test_sensitivity_rejects_unknown_selected_key():
    with pytest.raises(ValueError,match='Missing or duplicate'):
        compare_sensitivity(rfm_fixture(),pd.DataFrame({'key':['k1']}),{'kmeans':'missing'})
