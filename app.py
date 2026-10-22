
import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timedelta, timezone

from supabase import create_client
from itsdangerous import URLSafeTimedSerializer, BadSignature
from streamlit_cookies_controller import CookieController

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
# SUPABASE CONFIGURATION
# --------------------------------------------------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_SECRET_KEY = st.secrets["SUPABASE_SECRET_KEY"]
    SESSION_SIGNING_KEY = st.secrets["SESSION_SIGNING_KEY"]

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    session_db = create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )

    signer = URLSafeTimedSerializer(
        SESSION_SIGNING_KEY,
        salt="customer-segmentation-session",
    )

except Exception as exc:
    st.error(f"Supabase configuration error: {exc}")
    st.stop()


# --------------------------------------------------
# THREE-DAY LOGIN SESSION
# --------------------------------------------------
SESSION_SECONDS = 3 * 24 * 60 * 60
COOKIE_NAME = "customer_segmentation_session"

controller = CookieController(key="customer_session_cookie")
controller.getAll()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "logged_in": False,
    "otp_email": "",
    "auth_email": "",
    "cookie_checked": False,
    "saved_session_id": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# CREATE A LOGIN SESSION
# --------------------------------------------------
def create_login_session(user_id, email):
    session_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=3)

    # Save the session in Supabase.
    session_db.table("app_sessions").insert({
        "session_id": session_id,
        "user_id": user_id,
        "expires_at": expires_at.isoformat(),
    }).execute()

    # Sign the cookie data.
    token = signer.dumps({
        "session_id": session_id,
        "email": email,
    })

    # Save the cookie in the browser.
    controller.set(
        COOKIE_NAME,
        token,
        expires=expires_at,
        path="/",
        secure=True,
        same_site="lax",
    )

    st.session_state.logged_in = True
    st.session_state.auth_email = email
    st.session_state.saved_session_id = session_id


# --------------------------------------------------
# RESTORE A SAVED LOGIN
# --------------------------------------------------
def restore_login_session():
    token = controller.get(COOKIE_NAME)

    if not token:
        return False

    try:
        payload = signer.loads(
            token,
            max_age=SESSION_SECONDS,
        )

        session_id = payload.get("session_id")
        email = payload.get("email")

        if not session_id or not email:
            return False

        result = (
            session_db.table("app_sessions")
            .select("session_id, user_id, expires_at")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            return False

        session = result.data[0]

        expiry = datetime.fromisoformat(
            session["expires_at"].replace("Z", "+00:00")
        )

        if expiry <= datetime.now(timezone.utc):
            session_db.table("app_sessions").delete().eq(
                "session_id", session_id
            ).execute()
            return False

        st.session_state.logged_in = True
        st.session_state.auth_email = email
        st.session_state.saved_session_id = session_id

        return True

    except (BadSignature, ValueError, TypeError, KeyError):
        return False

    except Exception as exc:
        st.warning(f"Could not restore saved login: {exc}")
        return False


# --------------------------------------------------
# CHECK FOR A SAVED LOGIN
# --------------------------------------------------
if not st.session_state.logged_in:
    token = controller.get(COOKIE_NAME)

    # The cookie component may not have loaded yet.
    if token is None:
        st.info("Restoring your saved login...")
        st.stop()

    if token:
        restore_login_session()

    st.session_state.cookie_checked = True


# --------------------------------------------------
# EMAIL OTP LOGIN
# --------------------------------------------------
if not st.session_state.logged_in:

    st.title("🔐 Welcome!")
    st.subheader("Customer Segmentation & Predictive Analytics")
    st.write("Sign in using your email address.")

    with st.form("email_form"):
        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
        ).strip().lower()

        send_otp = st.form_submit_button(
            "Send verification code",
            use_container_width=True,
        )

    if send_otp:
        if (
            not email
            or "@" not in email
            or "." not in email.split("@")[-1]
        ):
            st.error("Please enter a valid email address.")

        else:
            try:
                supabase.auth.sign_in_with_otp({
                    "email": email,
                    "options": {
                        "should_create_user": True,
                    },
                })

                st.session_state.otp_email = email

                st.success(
                    "If the request succeeded, check your email "
                    "for the verification code."
                )

            except Exception as exc:
                st.error(f"Could not send verification code: {exc}")

    if st.session_state.otp_email:

        st.info(
            f"Enter the verification code sent to "
            f"{st.session_state.otp_email}"
        )

        with st.form("otp_form"):
            otp = st.text_input(
                "Email verification code",
                max_chars=10,
                placeholder="Enter the code from your email",
            ).strip()

            verify = st.form_submit_button(
                "Verify and Login",
                use_container_width=True,
            )

        if verify:
            if not otp.isdigit():
                st.error("Please enter the numeric verification code.")

            else:
                try:
                    response = supabase.auth.verify_otp({
                        "email": st.session_state.otp_email,
                        "token": otp,
                        "type": "email",
                    })

                    if not response.user:
                        st.error(
                            "Verification failed. "
                            "Please request a new code."
                        )

                    else:
                        create_login_session(
                            response.user.id,
                            st.session_state.otp_email,
                        )

                        st.session_state.otp_email = ""
                        st.session_state.cookie_checked = True

                        st.rerun()

                except Exception as exc:
                    # Display the real error for diagnosis.
                    st.error(
                        f"Login/session error: {type(exc).__name__}: {exc}"
                    )

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
    st.write(f"Signed in as: {st.session_state.auth_email}")

    if st.button("Logout", use_container_width=True):

        try:
            session_id = st.session_state.get("saved_session_id")

            if session_id:
                (
                    session_db.table("app_sessions")
                    .delete()
                    .eq("session_id", session_id)
                    .execute()
                )

            controller.remove(COOKIE_NAME, path="/")

            supabase.auth.sign_out()

        except Exception as exc:
            st.warning(f"Logout cleanup warning: {exc}")

        st.session_state.logged_in = False
        st.session_state.otp_email = ""
        st.session_state.auth_email = ""
        st.session_state.saved_session_id = None
        st.session_state.cookie_checked = True

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
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Customer Segmentation",
    "Predictive Analytics",
    "Data Explorer",
])


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

    top = rfm.sort_values("Monetary", ascending=False).head(10)

    st.dataframe(
        top[[
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "SegmentName",
        ]],
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

    st.dataframe(summary.round(2), use_container_width=True)

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

    st.dataframe(df.head(500), use_container_width=True)

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
