"""Feature preparation and deterministic preliminary K-Means experiments."""
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .rfm_analysis import FEATURES
from .evaluation import evaluate_clusters


def prepare_features(rfm):
    """Keep Recency in days; log1p Frequency/Monetary, then standardize all three."""
    transformed = rfm[FEATURES].astype(float).copy()
    transformed[["Frequency", "Monetary"]] = np.log1p(transformed[["Frequency", "Monetary"]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite scaled features")
    return scaled, scaler, transformed


def run_kmeans(features):
    """Fit K=2..10 where n and distinct points support internal metrics."""
    limit = min(10, len(features) - 1, len(np.unique(features, axis=0)))
    if limit < 2:
        raise ValueError("Need at least three customers and two distinct RFM points")
    rows, models, assignments = [], {}, {}
    for k in range(2, limit + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20, algorithm="lloyd")
        start = perf_counter()
        labels = model.fit_predict(features)
        elapsed = perf_counter() - start
        if len(np.unique(labels)) != k:
            raise ValueError(f"K={k} produced fewer than K clusters")
        rows.append({"k": k, "inertia": float(model.inertia_), "fit_seconds": elapsed,
                     **evaluate_clusters(features, labels),
                     **{f"cluster_{i}_size": int((labels == i).sum()) for i in range(k)}})
        models[k], assignments[k] = model, labels
    return pd.DataFrame(rows), models, assignments


def profile_segments(rfm, labels):
    """Summarize untreated RFM units and describe relative median behavior."""
    customers = rfm.assign(Cluster=labels)
    profile = customers.groupby("Cluster").agg(
        customers=("CustomerID", "size"),
        mean_recency=("Recency", "mean"), median_recency=("Recency", "median"),
        mean_frequency=("Frequency", "mean"), median_frequency=("Frequency", "median"),
        mean_monetary=("Monetary", "mean"), median_monetary=("Monetary", "median"))
    profile["percentage"] = 100 * profile.customers / len(customers)
    med = rfm[FEATURES].median()
    profile["description"] = profile.apply(lambda row: ", ".join([
        "more recent" if row.median_recency <= med.Recency else "less recent",
        "higher frequency" if row.median_frequency > med.Frequency else "lower frequency",
        "higher spending" if row.median_monetary > med.Monetary else "lower spending"])
        + " (relative to overall customer medians)", axis=1)
    return customers, profile.reset_index()


def select_k(comparison, profiles):
    """Transparent preliminary heuristic balancing separation, elbow, and usable sizes."""
    k = comparison.k.to_numpy()
    inertia = comparison.inertia.to_numpy()
    if len(k) == 1:
        elbow = int(k[0])
    else:
        x = (k-k.min())/(k.max()-k.min())
        y = (inertia-inertia.min())/(inertia.max()-inertia.min()) if inertia.max()>inertia.min() else np.zeros(len(k))
        elbow = int(k[np.argmax(1-x-y)])
    best = float(comparison.silhouette.max())
    candidates = comparison.loc[comparison.silhouette >= best - 0.03].copy()
    usable = [int(v) for v in candidates.k if profiles[int(v)].percentage.min() >= 5]
    pool = usable or candidates.k.astype(int).tolist()
    selected = min(pool, key=lambda v: (abs(v-elbow), v))
    rationale = (f"Preliminary K={selected}: candidates within 0.03 of the best silhouette ({best:.4f}); "
                 f"prefer profiles with all segments >=5% of customers, then nearest to the normalized "
                 f"inertia-curve elbow K={elbow}, breaking ties toward fewer segments. "
                 "Profiles describe differences in original RFM units; labels are relative descriptions, not demonstrated churn. "
                 "This transparent heuristic is provisional, not proof of optimal business usefulness.")
    return selected, rationale, elbow
