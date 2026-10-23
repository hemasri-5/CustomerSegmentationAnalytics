
"""RFM feature engineering, K-Means segmentation, and cluster evaluation."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_rfm(df):
    """Create customer-level Recency, Frequency, and Monetary features."""
    if df.empty:
        raise ValueError("The transaction dataset is empty.")

    required = {"CustomerID", "InvoiceDate", "InvoiceNo", "Sales", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = df.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
    if data.empty:
        raise ValueError("No valid customer transactions were found.")

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    if data.empty:
        raise ValueError("No valid invoice dates were found.")

    snapshot_date = data["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = data.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
        TotalQuantity=("Quantity", "sum"),
    )

    rfm["Recency"] = (
        snapshot_date - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    return rfm.reset_index()


def _prepare_features(rfm):
    """Log-transform and standardize RFM features consistently."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0).clip(lower=0)
    features["Frequency"] = features["Frequency"].clip(lower=1)
    features["Monetary"] = features["Monetary"].clip(lower=0.01)

    transformed = np.log1p(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)

    return scaled


def evaluate_clusters(df, max_clusters=8):
    """
    Evaluate candidate K values using inertia and silhouette score.

    Returns a DataFrame with K, Inertia, and SilhouetteScore.
    The caller can use this result to display evaluation charts.
    """
    rfm = build_rfm(df)
    customer_count = len(rfm)

    if customer_count < 3:
        raise ValueError(
            "At least 3 distinct customers are needed to evaluate clusters."
        )

    X_scaled = _prepare_features(rfm)
    upper_k = min(max(2, int(max_clusters)), customer_count - 1)

    results = []

    for k in range(2, upper_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(X_scaled)

        # Silhouette score is undefined if clustering produces
        # fewer than 2 distinct labels or one label per customer.
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= customer_count:
            score = np.nan
        else:
            try:
                score = float(silhouette_score(X_scaled, labels))
            except ValueError:
                score = np.nan

        results.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "SilhouetteScore": score,
        })

    return pd.DataFrame(results)


def segment_customers(df, n_clusters=4):
    """Return customer-level RFM data and a segment summary."""
    rfm = build_rfm(df)

    if rfm.empty:
        raise ValueError("No customers are available for segmentation.")

    X_scaled = _prepare_features(rfm)
    k = max(1, min(int(n_clusters), len(rfm)))

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    rfm["Segment"] = model.fit_predict(X_scaled)

    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "nunique"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).reset_index()

    
    # Assign more descriptive labels based on segment characteristics.
    spend_median = summary["AvgMonetary"].median()
    freq_median = summary["AvgFrequency"].median()
    recency_median = summary["AvgRecency"].median()

    labels = {}

    for _, row in summary.iterrows():
        recency = row["AvgRecency"]
        frequency = row["AvgFrequency"]
        monetary = row["AvgMonetary"]

        if (
            monetary >= spend_median
            and frequency >= freq_median
            and recency <= recency_median
        ):
            label = "Active high-value customers"

        elif (
            recency > recency_median
            and frequency < freq_median
        ):
            label = "At-risk / needs re-engagement"

        elif (
            recency <= recency_median
            and frequency >= freq_median
        ):
            label = "Recent frequent buyers"

        elif monetary < spend_median:
            label = "Lower-spend customers"

        else:
            label = "Developing customer relationships"

        labels[row["Segment"]] = label

    rfm["SegmentName"] = rfm["Segment"].map(labels)
    summary["SegmentName"] = summary["Segment"].map(labels)

    return rfm, summary.sort_values("Segment").reset_index(drop=True)
