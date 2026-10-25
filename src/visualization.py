"""Report figures with original units and explicit projection caveats."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from .rfm_analysis import FEATURES
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def eda_figures(raw, directory):
    fig, ax = plt.subplots(figsize=(9, 5))
    (raw.isna().mean()*100).plot.bar(ax=ax)
    ax.set(title="Missing values in original transactions", xlabel="Column", ylabel="Missing records (%)")
    save(fig, directory, "missing_values")
    fig, ax = plt.subplots(figsize=(9, 5))
    raw.groupby(raw.InvoiceDate.dt.to_period("M")).size().plot.bar(ax=ax)
    ax.set(title="Monthly original transaction volume", xlabel="Month", ylabel="Transaction rows")
    save(fig, directory, "monthly_transactions")
    fig, ax = plt.subplots(figsize=(9, 5))
    raw.Country.value_counts().head(10).plot.bar(ax=ax)
    ax.set(title="Top countries by original transaction volume", xlabel="Country", ylabel="Transaction rows")
    save(fig, directory, "top_countries")
    fig, ax = plt.subplots(figsize=(9, 5))
    values = raw.Quantity * raw.UnitPrice
    values = values[np.isfinite(values)]
    ax.hist(values, bins=150)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set(title="Original line-item transaction values (including returns)", xlabel="Value (£ GBP, symmetric log scale)", ylabel="Transaction rows (log scale)")
    save(fig, directory, "transaction_values")


def rfm_figures(rfm, directory):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feature, unit in zip(axes, FEATURES, ["days", "distinct invoices", "£ GBP"]):
        ax.hist(rfm[feature], bins=60)
        ax.set(title=feature, xlabel=f"{feature} ({unit})", ylabel="Customers")
        ax.set_yscale("log")
    fig.suptitle("Untreated RFM distributions (all customers retained)")
    save(fig, directory, "rfm_distributions")


def clustering_figures(features, labels, comparison, profile, directory):
    for metric, title, ylabel in [("inertia", "K-Means Elbow Method", "Within-cluster sum of squares"),
                                  ("silhouette", "Silhouette versus K", "Silhouette Score (higher is better)"),
                                  ("davies_bouldin", "Davies–Bouldin versus K", "Davies–Bouldin Index (lower is better)")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(comparison.k, comparison[metric], "o-")
        ax.set(title=title, xlabel="Number of clusters (K)", ylabel=ylabel, xticks=comparison.k)
        save(fig, directory, metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(profile.Cluster.astype(str), profile.customers)
    ax.set(title="Selected K-Means segment sizes", xlabel="Cluster ID", ylabel="Customers")
    save(fig, directory, "cluster_sizes")
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in sorted(set(labels)):
        ax.scatter(*projected[labels == label].T, s=12, alpha=.55, label=f"Cluster {label}")
    ax.legend()
    ax.set(title="PCA visualization only; clustering uses all three scaled RFM features",
           xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    save(fig, directory, "pca_segments")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feature, unit in zip(axes, ["recency", "frequency", "monetary"], ["days", "invoices", "£ GBP"]):
        positions = np.arange(len(profile))
        ax.bar(positions-.2, profile[f"mean_{feature}"], .4, label="Mean")
        ax.bar(positions+.2, profile[f"median_{feature}"], .4, label="Median")
        ax.set(title=feature.title(), xlabel="Cluster ID", ylabel=unit, xticks=positions, xticklabels=profile.Cluster)
        ax.legend()
    fig.suptitle("Cluster RFM profiles in original units")
    save(fig, directory, "cluster_rfm")
    return pca.explained_variance_ratio_.tolist()
