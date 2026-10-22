"""RFM feature engineering and K-Means customer segmentation."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def build_rfm(df):
    """Create Recency, Frequency and Monetary metrics at customer level."""
    snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Sales", "sum"),
        TotalQuantity=("Quantity", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"].dt.normalize()).dt.days
    rfm = rfm.reset_index()
    return rfm

def segment_customers(df, n_clusters=4):
    rfm = build_rfm(df)
    features = ["Recency", "Frequency", "Monetary"]
    # Log transform skewed frequency/spending; then scale before K-Means.
    model_features = rfm[features].clip(lower=0).copy()
    model_features["Frequency"] = model_features["Frequency"].clip(lower=1)
    model_features["Monetary"] = model_features["Monetary"].clip(lower=0.01)
    import numpy as np
    X = np.log1p(model_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = max(1, min(int(n_clusters), len(rfm)))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["Segment"] = model.fit_predict(X_scaled)
    # Generate human-readable labels based on relative group spend/frequency/recency.
    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).reset_index()
    if len(summary):
        spend_median = summary["AvgMonetary"].median()
        freq_median = summary["AvgFrequency"].median()
        for _, row in summary.iterrows():
            if row["AvgMonetary"] >= spend_median and row["AvgFrequency"] >= freq_median:
                label = "High-value / loyal"
            elif row["AvgRecency"] <= summary["AvgRecency"].median():
                label = "Recently active"
            elif row["AvgMonetary"] < spend_median:
                label = "Budget / low spend"
            else:
                label = "Needs engagement"
            rfm.loc[rfm["Segment"] == row["Segment"], "SegmentName"] = label
        summary["SegmentName"] = summary["Segment"].map(
            rfm.drop_duplicates("Segment").set_index("Segment")["SegmentName"].to_dict()
        )
    return rfm, summary.sort_values("Segment")
