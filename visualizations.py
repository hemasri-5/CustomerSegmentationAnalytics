"""Reusable chart functions for the Streamlit dashboard."""
import matplotlib.pyplot as plt
import seaborn as sns

def segment_scatter(rfm):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="SegmentName",
                    size="Recency", sizes=(25, 180), alpha=.8, ax=ax)
    ax.set_title("Customer Segments: Frequency vs Monetary Value")
    ax.set_xlabel("Number of distinct orders")
    ax.set_ylabel("Total customer spend")
    fig.tight_layout()
    return fig

def segment_bar(summary):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=summary, x="SegmentName", y="Customers", hue="SegmentName",
                legend=False, ax=ax)
    ax.set_title("Customers in Each Segment")
    ax.set_xlabel("")
    ax.set_ylabel("Number of customers")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig

def sales_line(sales):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sales["InvoiceDate"], sales["Sales"], marker="o", linewidth=1.5)
    ax.set_title("Historical Weekly Sales")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    fig.tight_layout()
    return fig

def forecast_line(evaluation_data, forecasts):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not evaluation_data.empty:
        ax.plot(evaluation_data["InvoiceDate"], evaluation_data["Sales"], marker="o", label="Actual (test)")
        ax.plot(evaluation_data["InvoiceDate"], evaluation_data["PredictedSales"], marker="o", label="Predicted (test)")
    if not forecasts.empty:
        ax.plot(forecasts["InvoiceDate"], forecasts["PredictedSales"], marker="o", linestyle="--", label="Future forecast")
    ax.set_title("Sales Forecast and Model Evaluation")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    return fig
