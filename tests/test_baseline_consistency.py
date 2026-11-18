import numpy as np
import pandas as pd
import pytest
from src.final_research import features_for, fit_algorithm, metrics
from part3_sensitivity import compare_sensitivity

@pytest.mark.parametrize("algorithm,params", [
    ("kmeans", {"k": 2}),
    ("hierarchical", {"k": 3, "linkage": "average"}),
    ("dbscan", {"eps": .6, "min_samples": 4}),
])
def test_baseline_matches_selected_model_evaluation(algorithm, params):
    rng = np.random.default_rng(42)
    rfm = pd.DataFrame({
        "CustomerID": [str(i) for i in range(150)],
        "Recency": np.r_[rng.integers(1,30,50), rng.integers(70,150,50), rng.integers(180,300,50)],
        "Frequency": np.r_[rng.integers(8,20,50), rng.integers(3,8,50), rng.integers(1,4,50)],
        "Monetary": np.r_[rng.integers(1200,5000,50), rng.integers(200,1100,50), rng.integers(20,300,50)],
    })
    labels, _ = fit_algorithm(features_for(rfm), algorithm, params)
    expected = metrics(features_for(rfm), labels)
    key = algorithm + "_test"
    grid = pd.DataFrame([{"key": key, "algorithm": algorithm, **params}])
    actual = compare_sensitivity(rfm, grid, {algorithm: key})
    baseline = actual.loc[actual.variant == "baseline"].iloc[0]
    for name in ("silhouette", "davies_bouldin", "calinski_harabasz", "coverage"):
        if np.isnan(expected[name]):
            assert np.isnan(baseline[name])
        else:
            assert baseline[name] == pytest.approx(expected[name], abs=1e-10)
