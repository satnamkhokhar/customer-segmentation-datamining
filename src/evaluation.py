"""Internal clustering quality measures in the modeled feature space."""
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clusters(features, labels):
    """Evaluate all customers; no subsampling is used for silhouette."""
    return {"silhouette": float(silhouette_score(features, labels)),
            "davies_bouldin": float(davies_bouldin_score(features, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(features, labels))}
