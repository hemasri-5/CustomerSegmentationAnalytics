
import streamlit as st
import pandas as pd

from preprocessing import load_transactions
from segmentation import segment_customers
from prediction import forecast_sales
from visualizations import (
    segment_scatter,
    segment_bar,
    sales_line,
    forecast_line,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Predictive Analytics",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# DEMO LOGIN SYSTEM
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Welcome Back!")
    st.subheader("Customer Segmentation & Predictive Analytics")
    st.write("Please log in to access your dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True,
        )

    if submitted:
        if username == "admin" and password == "demo123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

    st.stop()

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
st.title("📊 Customer Segmentation & Predictive Analytics System")
st.caption(
    "A team project combining customer grouping, "
    "sales forecasting, and business insights."
)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.header("Project Controls")

    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    st.divider()

    uploaded = st.file_uploader(
        "Upload transaction data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
    )

    clusters = st.slider(
        "Number of customer segments",
        min_value=2,
        max_value=8,
        value=4,
    )

    periods = st.slider(
        "Weeks to forecast",
        min_value=1,
        max_value=12,
        value=4,
    )

    st.caption(
        "Required columns: InvoiceNo, Quantity, InvoiceDate, "
        "UnitPrice, CustomerID."
    )
    st.caption(
        "If no file is uploaded, the app uses the included "
        "synthetic demo dataset."
    )

# --------------------------------------------------
# LOAD TRANSACTION DATA
# --------------------------------------------------
try:
    df = load_transactions(uploaded_file=uploaded)

except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.stop()

# --------------------------------------------------
# CUSTOMER SEGMENTATION AND SALES FORECASTING
# --------------------------------------------------
with st.spinner(
    "Analyzing customers and training the forecasting model..."
):
    try:
        rfm, summary = segment_customers(df, clusters)

    except Exception as exc:
        st.error(f"Segmentation failed: {exc}")
        st.stop()

    try:
        sales, evaluation_data, forecasts, metrics = forecast_sales(
            df, periods
        )
        prediction_error = None

    except Exception as exc:
        sales = pd.DataFrame()
        evaluation_data = pd.DataFrame()
        forecasts = pd.DataFrame()
        metrics = {}
        prediction_error = str(exc)

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions / lines", f"{len(df):,}")
c2.metric("Unique customers", f"{df['CustomerID'].nunique():,}")
c3.metric("Total sales", f"{df['Sales'].sum():,.2f}")
c4.metric("Customer segments", f"{rfm['Segment'].nunique()}")

# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview",
        "Customer Segmentation",
        "Predictive Analytics",
        "Data Explorer",
    ]
)

# --------------------------------------------------
# TAB 1: BUSINESS OVERVIEW
# --------------------------------------------------
with tab1:
    st.subheader("Business Overview")

    st.write(
        "Use these insights to understand customer value "
        "and plan marketing or inventory decisions."
    )

    left, right = st.columns(2)

    with left:
        if not sales.empty:
            fig = sales_line(sales)

            if fig is not None:
                st.pyplot(fig, clear_figure=True)
        else:
            st.info("Not enough sales history to plot.")

    with right:
        fig = segment_bar(summary)

        if fig is not None:
            st.pyplot(fig, clear_figure=True)

    st.subheader("Top 10 Customers by Revenue")

    top = rfm.sort_values(
        "Monetary",
        ascending=False,
    ).head(10)

    st.dataframe(
        top[
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "SegmentName",
            ]
        ],
        use_container_width=True,
    )

# --------------------------------------------------
# TAB 2: CUSTOMER SEGMENTATION
# --------------------------------------------------
with tab2:
    st.subheader("Customer Segmentation with RFM + K-Means")

    st.write(
        "Recency = days since last purchase. "
        "Frequency = distinct orders. "
        "Monetary = total customer spend."
    )

    fig = segment_scatter(rfm)

    if fig is not None:
        st.pyplot(fig, clear_figure=True)

    st.subheader("Segment Summary")

    st.dataframe(
        summary.round(2),
        use_container_width=True,
    )

    st.download_button(
        "Download customer segments CSV",
        rfm.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download segment summary CSV",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="segment_summary.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# TAB 3: PREDICTIVE ANALYTICS
# --------------------------------------------------
with tab3:
    st.subheader("Weekly Sales Forecasting")

    if prediction_error:
        st.warning(
            f"Forecasting could not be completed: {prediction_error}"
        )

    elif not forecasts.empty and metrics:
        m1, m2, m3 = st.columns(3)

        m1.metric("MAE", f"{metrics['MAE']:,.2f}")
        m2.metric("RMSE", f"{metrics['RMSE']:,.2f}")
        m3.metric("Test weeks", f"{metrics['Test weeks']}")

        st.caption(
            "MAE and RMSE measure test-set error; lower is "
            "generally better. R² can be unstable on small "
            "or noisy test sets."
        )

        fig = forecast_line(evaluation_data, forecasts)

        if fig is not None:
            st.pyplot(fig, clear_figure=True)

        st.subheader("Future Sales Forecast")

        forecast_display = forecasts.rename(
            columns={
                "InvoiceDate": "Week starting",
                "PredictedSales": "Predicted sales",
            }
        )

        st.dataframe(
            forecast_display.round(2),
            use_container_width=True,
        )

        st.download_button(
            "Download forecast CSV",
            forecasts.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv",
        )

        if "R2" in metrics and pd.notna(metrics["R2"]):
            st.write(f"**Test-set R²:** {metrics['R2']:.3f}")
        else:
            st.write(
                "**Test-set R²:** not available for a single test week."
            )

        st.info(
            "Forecasts are estimates, not guarantees. Accuracy "
            "depends on data quality, history length, seasonality, "
            "and the forecasting method."
        )

    else:
        st.warning(
            "Forecast results are not available for this dataset."
        )

# --------------------------------------------------
# TAB 4: DATA EXPLORER
# --------------------------------------------------
with tab4:
    st.subheader("Cleaned Transaction Data")

    st.dataframe(
        df.head(500),
        use_container_width=True,
    )

    st.download_button(
        "Download cleaned data CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_transactions.csv",
        mime="text/csv",
    )

    st.caption("Showing up to the first 500 rows in the table.")

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------
st.divider()
st.subheader("Business Recommendations")

st.markdown("""
- **High-value / loyal customers:** Consider loyalty rewards
  and personalized recommendations.
- **Recently active customers:** Encourage repeat purchases
  with relevant offers.
- **Budget / low-spend customers:** Test affordable bundles
  and suitable offers.
- **Needs engagement:** Test re-engagement messages without
  excessive discounts.
- **Sales forecasting:** Use forecasts as one input for
  inventory and staffing decisions, and monitor model errors.
""")

st.caption(
    "Demo data is synthetic and for development/testing only. "
    "Replace it with a suitable real dataset before making "
    "real business decisions."
)
