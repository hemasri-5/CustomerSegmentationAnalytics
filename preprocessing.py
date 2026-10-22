"""Data loading and cleaning for the Customer Segmentation & Predictive Analytics System."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]

def load_transactions(uploaded_file=None, sample_path=None):
    """Load CSV/XLSX data, or the included demo CSV when no file is supplied."""
    if uploaded_file is not None:
        name = getattr(uploaded_file, "name", "").lower()
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    else:
        path = Path(sample_path) if sample_path else Path(__file__).with_name("sample_transactions.csv")
        df = pd.read_csv(path)

    # Normalize column labels, while retaining the common UCI Online Retail names.
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing) +
            ". Required columns are InvoiceNo, Quantity, InvoiceDate, UnitPrice, CustomerID."
        )

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=False)
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"])
    df = df.drop_duplicates()
    # Standard retail datasets may include cancellations/returns; exclude these for the
    # basic sales forecasting demonstration, and document this assumption.
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if df.empty:
        raise ValueError("No valid transactions remain after cleaning. Check the uploaded dataset.")
    return df.sort_values("InvoiceDate").reset_index(drop=True)
