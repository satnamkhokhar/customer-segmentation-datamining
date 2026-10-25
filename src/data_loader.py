"""Load and inspect transactions without filtering records."""
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np

REQUIRED = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download_dataset(path):
    """Download the official UCI archive using standard TLS verification."""
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(URL, timeout=120) as response:
            archive = ZipFile(BytesIO(response.read()))
    except URLError as exc:
        raise RuntimeError("Official UCI download failed. Ensure archive.ics.uci.edu is allowed in environment settings, or manually place Online Retail.xlsx in data/raw/. " + str(exc)) from exc
    members = [name for name in archive.namelist() if name.endswith("Online Retail.xlsx")]
    if len(members) != 1:
        raise ValueError("Official archive does not contain the expected workbook")
    path.write_bytes(archive.read(members[0]))


def validate_columns(df):
    """Reject incomplete input schemas."""
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_data(path):
    """Load Excel, validate numeric fields, and flag invalid dates as NaT."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}. Download Online Retail.xlsx from https://archive.ics.uci.edu/dataset/352/online+retail or run python main.py --download")
    df = pd.read_excel(path, engine="openpyxl")
    validate_columns(df)
    for column in ["Quantity", "UnitPrice"]:
        converted = pd.to_numeric(df[column], errors="coerce")
        if (df[column].notna() & converted.isna()).any():
            raise ValueError(f"Non-numeric values in {column}")
        df[column] = converted
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def explore(df):
    """Return JSON-compatible initial data quality and distribution statistics."""
    return {
        "rows": len(df), "columns": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_counts": df.isna().sum().to_dict(),
        "missing_percent": (100 * df.isna().mean()).to_dict(),
        "unique_customers": int(df.CustomerID.nunique()),
        "unique_invoices": int(df.InvoiceNo.nunique()),
        "cancelled_invoices": int(df.loc[df.InvoiceNo.astype(str).str.upper().str.startswith("C"), "InvoiceNo"].nunique()),
        "cancelled_rows": int(df.InvoiceNo.astype(str).str.upper().str.startswith("C").sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_quantity_rows": int((df.Quantity < 0).sum()),
        "invalid_price_rows": int((~np.isfinite(df.UnitPrice) | (df.UnitPrice <= 0)).sum()),
        "invalid_dates": int(df.InvoiceDate.isna().sum()),
        "date_min": str(df.InvoiceDate.min()), "date_max": str(df.InvoiceDate.max()),
        "purchase_statistics": df[["Quantity", "UnitPrice"]].describe().to_dict(),
        "top_countries": df.Country.value_counts().head(10).to_dict(),
        "monthly_transactions": df.groupby(df.InvoiceDate.dt.to_period("M")).size().rename(index=str).to_dict(),
    }
