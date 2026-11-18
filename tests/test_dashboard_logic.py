import numpy as np
import pandas as pd
import pytest
from dashboard_logic import weighted_scores,run_live,read_saved


def sample_rfm():
    rng=np.random.default_rng(42)
    return pd.DataFrame({'CustomerID':[str(i) for i in range(80)],
                         'Recency':np.r_[rng.integers(1,20,40),rng.integers(100,200,40)],
                         'Frequency':np.r_[rng.integers(5,15,40),rng.integers(1,3,40)],
                         'Monetary':np.r_[rng.uniform(1000,5000,40),rng.uniform(30,200,40)]})

@pytest.mark.parametrize('algorithm,params',[('kmeans',{'k':2}),('hierarchical',{'k':2}),('dbscan',{'eps':.7,'min_samples':3})])
def test_live_models(algorithm,params):
    metrics,seconds,profiles,scatter=run_live(sample_rfm(),algorithm,**params)
    assert len(scatter)==80 and sum(profiles.customers)==80
    assert 0<=metrics['coverage']<=1 and seconds>=0
    assert {'PC1','PC2','Cluster','CustomerID'}<=set(scatter)

def test_weights_and_validation():
    scores=pd.DataFrame({'algorithm':['a','b'],'Q':[1,0],'S':[0,1],'C':[1,1]})
    assert weighted_scores(scores).iloc[0].algorithm=='a'
    assert weighted_scores(scores,.1,.8,.1).iloc[0].algorithm=='b'
    with pytest.raises(ValueError):weighted_scores(scores,.8,.8,-.6)
    with pytest.raises(ValueError):read_saved('../requirements.txt')
