"""Sequential, auditable purchase cleaning."""
import numpy as np
import pandas as pd
from .data_loader import validate_columns


def clean_transactions(df):
    """Return cleaned purchases and a reconciliable sequential removal ledger."""
    validate_columns(df)
    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data.InvoiceDate, errors="coerce")
    for field in ["Quantity", "UnitPrice"]:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    steps = []
    def remove(name, mask):
        nonlocal data
        before = len(data)
        data = data.loc[~mask].copy()
        removed = before - len(data)
        steps.append({"step": name, "rows_before": before, "rows_removed": removed,
                      "rows_remaining": len(data), "percentage_removed": 100 * removed / before if before else 0})
    remove("missing_customer_id", data.CustomerID.isna() | data.CustomerID.astype(str).str.strip().eq(""))
    remove("exact_duplicates", data.duplicated())
    remove("cancelled_invoices", data.InvoiceNo.astype(str).str.strip().str.upper().str.startswith("C"))
    remove("invalid_quantity", ~np.isfinite(data.Quantity) | (data.Quantity <= 0))
    remove("invalid_price", ~np.isfinite(data.UnitPrice) | (data.UnitPrice <= 0))
    remove("invalid_date", data.InvoiceDate.isna())
    remove("missing_invoice", data.InvoiceNo.isna() | data.InvoiceNo.astype(str).str.strip().eq(""))
    data["CustomerID"] = data.CustomerID.map(lambda value: str(int(value)) if isinstance(value, (float, int)) and float(value).is_integer() else str(value).strip())
    data["InvoiceNo"] = data.InvoiceNo.astype(str).str.strip()
    data["TotalPrice"] = data.Quantity * data.UnitPrice
    if not np.isfinite(data.TotalPrice).all() or not (data.TotalPrice > 0).all():
        raise ValueError("Invalid purchase totals after cleaning")
    if data.empty:
        raise ValueError("No valid purchases remain after cleaning")
    return data, pd.DataFrame(steps)
