"""Aggregate purchases into customer-level RFM features."""
import numpy as np
import pandas as pd
FEATURES = ["Recency", "Frequency", "Monetary"]


def calculate_rfm(transactions):
    """Use calendar-day recency relative to one day after the latest purchase."""
    reference = transactions.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    grouped = transactions.groupby("CustomerID")
    rfm = grouped.agg(LastPurchase=("InvoiceDate", "max"), Frequency=("InvoiceNo", "nunique"), Monetary=("TotalPrice", "sum"))
    rfm["Recency"] = (reference - rfm.LastPurchase.dt.normalize()).dt.days
    rfm = rfm[FEATURES].reset_index()
    values = rfm[FEATURES].to_numpy()
    if (not rfm.CustomerID.is_unique or not np.isfinite(values).all()
            or (rfm.Recency < 0).any() or (rfm[["Frequency", "Monetary"]] <= 0).any().any()):
        raise ValueError("Invalid RFM features")
    return rfm, reference
