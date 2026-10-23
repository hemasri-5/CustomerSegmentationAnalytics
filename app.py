import time
import streamlit_url_fragment
from urllib.parse import parse_qs
import streamlit as st
import pandas as pd
import uuid
from datetime import datetime, timedelta, timezone

from supabase import create_client
from itsdangerous import URLSafeTimedSerializer, BadSignature
from streamlit_cookies_controller import CookieController

from preprocessing import load_transactions
from segmentation import segment_customers, evaluate_clusters
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
    session_db = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

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


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "logged_in": False,
    "otp_email": "",
    "auth_email": "",
    "saved_session_id": None,
    "login_restore_attempted": False,
    "recovery_mode": False,
    "recovery_completed": False,
    "recovery_error": "",
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

    session_db.table("app_sessions").insert({
        "session_id": session_id,
        "user_id": user_id,
        "expires_at": expires_at.isoformat(),
    }).execute()

    token = signer.dumps({
        "session_id": session_id,
        "email": email,
    })

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
def restore_login_session(token):
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
            (
                session_db.table("app_sessions")
                .delete()
                .eq("session_id", session_id)
                .execute()
            )
            controller.remove(COOKIE_NAME, path="/")
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
# PASSWORD RECOVERY
# --------------------------------------------------
fragment = streamlit_url_fragment.get_fragment()

if fragment and not st.session_state.recovery_completed:
    params = parse_qs(fragment.lstrip("#"))

    recovery_type = params.get("type", [""])[0]
    access_token = params.get("access_token", [""])[0]
    refresh_token = params.get("refresh_token", [""])[0]

    if recovery_type == "recovery":
        if access_token and refresh_token:
            try:
                supabase.auth.set_session(
                    access_token,
                    refresh_token,
                )
                st.session_state.recovery_mode = True

            except Exception:
                st.session_state.recovery_error = (
                    "Your reset link is invalid or expired. "
                    "Please request a new reset email."
                )
        else:
            st.session_state.recovery_error = (
                "Recovery information is missing. "
                "Please request a new reset email."
            )

if st.session_state.recovery_error:
    st.title("Reset your password")
    st.error(st.session_state.recovery_error)
    st.stop()

if st.session_state.recovery_mode:
    st.title("🔐 Set a New Password")

    with st.form("set_new_password_form"):
        new_password = st.text_input(
            "New password",
            type="password",
        )
        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Update Password",
            use_container_width=True,
        )

    if submitted:
        if len(new_password) < 8:
            st.error("Use at least 8 characters.")
        elif new_password != confirm_password:
            st.error("The passwords do not match.")
        else:
            try:
                response = supabase.auth.update_user({
                    "password": new_password,
                })

                if response.user:
                    st.session_state.recovery_mode = False
                    st.session_state.recovery_completed = True
                    st.success(
                        "Password updated! You can now log in "
                        "with your new password."
                    )
                    st.rerun()
                else:
                    st.error("Password update failed.")

            except Exception:
                st.error(
                    "Could not update the password. "
                    "Request a new reset link and try again."
                )

    st.stop()

# --------------------------------------------------
# CHECK FOR A SAVED LOGIN
# --------------------------------------------------
# CHECK FOR A SAVED LOGIN
if not st.session_state.logged_in:
    cookies = controller.getAll()

    if cookies is None:
        with st.spinner("Restoring your saved login..."):
            time.sleep(1)
        cookies = controller.getAll()

    if cookies is None:
        st.error(
            "The browser cookie component did not respond. "
            "Refresh the page and try again."
        )
        st.stop()

    token = cookies.get(COOKIE_NAME)

    if token:
        restore_login_session(token)

# --------------------------------------------------
# EMAIL OTP LOGIN
# --------------------------------------------------

# --------------------------------------------------
# LOGIN AND SIGN UP
# --------------------------------------------------

# --------------------------------------------------
# LOGIN, SIGN UP AND FORGOT PASSWORD
# --------------------------------------------------
if not st.session_state.logged_in:
    st.title("🔐 Welcome!")
    st.subheader("Customer Segmentation & Predictive Analytics")

    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "Login"

    login_tab, signup_tab, forgot_tab = st.tabs(
        ["Login", "Sign Up", "Forgot Password?"]
    )

    # LOGIN WITH EMAIL AND PASSWORD
    with login_tab:
        st.write("Login with your registered email and password.")

        with st.form("password_login_form"):
            email = st.text_input(
                "Email address",
                key="login_email",
            ).strip().lower()

            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_button = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_button:
            if not email or not password:
                st.error("Enter your email and password.")
            else:
                try:
                    response = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password,
                    })

                    if response.user:
                        create_login_session(response.user.id, email)
                        st.rerun()
                    else:
                        st.error("Login failed. Check your details.")

                except Exception:
                    st.error(
                        "Login failed. Check your email and password, "
                        "or use Forgot Password if needed."
                    )

    # SIGN UP WITH EMAIL AND PASSWORD
    with signup_tab:
        st.write("Create a new account.")

        with st.form("password_signup_form"):
            new_email = st.text_input(
                "Email address",
                key="signup_email",
            ).strip().lower()

            new_password = st.text_input(
                "Create password",
                type="password",
                key="signup_password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_signup_password",
            )

            signup_button = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if signup_button:
            if not new_email or not new_password:
                st.error("Enter your email and password.")
            elif "@" not in new_email:
                st.error("Enter a valid email address.")
            elif len(new_password) < 8:
                st.error("Use a password with at least 8 characters.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": new_email,
                        "password": new_password,
                    })

                    if response.user and response.session:
                        create_login_session(
                            response.user.id,
                            new_email,
                        )
                        st.success("Account created successfully!")
                        st.rerun()
                    elif response.user:
                        st.success(
                            "Sign-up request received. Check your email "
                            "to confirm your account, then log in."
                        )
                    else:
                        st.info(
                            "If sign-up succeeds, check your email "
                            "for confirmation instructions."
                        )

                except Exception:
                    st.error(
                        "Could not create the account. Check the email "
                        "or try logging in if you already registered."
                    )

    # FORGOT PASSWORD
    with forgot_tab:
        st.write("Request an email link to reset your password.")

        with st.form("forgot_password_form"):
            reset_email = st.text_input(
                "Registered email address",
                key="reset_email",
            ).strip().lower()

            reset_button = st.form_submit_button(
                "Send Password Reset Email",
                use_container_width=True,
            )

        if reset_button:
            if not reset_email or "@" not in reset_email:
                st.error("Enter a valid email address.")
            else:
                try:
                    
                    supabase.auth.reset_password_for_email(
                        reset_email,
                        {
                            "redirect_to": "https://customersegmentationanalytics-dx7lqqo7ziyxfgi9rzw5dq.streamlit.app/"
                        },
                    )

                    st.success(
                        "If an account exists for that email, "
                        "password-reset instructions will be sent. "
                        "Check your inbox."
                    )
                except Exception:
                    st.error(
                        "Could not send the reset email. "
                        "Please try again later."
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
# TAB 2: ADVANCED CUSTOMER SEGMENTATION DASHBOARD
# --------------------------------------------------
with tab2:
    st.subheader("Advanced Customer Segmentation Dashboard")
    st.write(
        "Explore customer value, buying activity, segment profiles, "
        "and customer-level details using RFM analysis and K-Means."
    )

    required_rfm_columns = {
        "CustomerID", "Recency", "Frequency", "Monetary",
        "Segment", "SegmentName"
    }
    missing_rfm_columns = required_rfm_columns - set(rfm.columns)

    if missing_rfm_columns:
        st.error(
            "The segmentation output is missing required columns: "
            + ", ".join(sorted(missing_rfm_columns))
        )
    else:
        # Normalize values for robust dashboard filters.
        dashboard_rfm = rfm.copy()
        dashboard_rfm["CustomerID"] = dashboard_rfm["CustomerID"].astype(str)
        dashboard_rfm["SegmentName"] = (
            dashboard_rfm["SegmentName"].fillna("Unlabelled segment").astype(str)
        )

        st.markdown("### Customer portfolio at a glance")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers analyzed", f"{dashboard_rfm['CustomerID'].nunique():,}")
        k2.metric("Total customer revenue", f"{dashboard_rfm['Monetary'].sum():,.2f}")
        k3.metric("Average customer spend", f"{dashboard_rfm['Monetary'].mean():,.2f}")
        k4.metric("Average order frequency", f"{dashboard_rfm['Frequency'].mean():,.2f}")

        st.divider()
        st.markdown("### Interactive customer filters")

        f1, f2, f3 = st.columns([1, 1, 1.4])
        segment_options = ["All segments"] + sorted(
            dashboard_rfm["SegmentName"].unique().tolist()
        )

        with f1:
            selected_segment_name = st.selectbox(
                "Customer segment",
                options=segment_options,
                key="advanced_segment_filter",
            )

        with f2:
            sort_choice = st.selectbox(
                "Sort customers by",
                options=[
                    "Highest spend",
                    "Lowest spend",
                    "Most frequent orders",
                    "Least recent purchase",
                    "Most recent purchase",
                ],
                key="advanced_segment_sort",
            )

        with f3:
            customer_search = st.text_input(
                "Search Customer ID",
                placeholder="Enter all or part of a customer ID",
                key="advanced_customer_search",
            ).strip()

        filtered_rfm = dashboard_rfm.copy()
        if selected_segment_name != "All segments":
            filtered_rfm = filtered_rfm[
                filtered_rfm["SegmentName"] == selected_segment_name
            ]

        if customer_search:
            filtered_rfm = filtered_rfm[
                filtered_rfm["CustomerID"].str.contains(
                    customer_search, case=False, na=False, regex=False
                )
            ]

        sort_columns = {
            "Highest spend": ("Monetary", False),
            "Lowest spend": ("Monetary", True),
            "Most frequent orders": ("Frequency", False),
            "Least recent purchase": ("Recency", False),
            "Most recent purchase": ("Recency", True),
        }
        sort_column, ascending = sort_columns[sort_choice]
        filtered_rfm = filtered_rfm.sort_values(
            sort_column, ascending=ascending
        )

        st.caption(
            f"Showing {len(filtered_rfm):,} customer record(s) after filters."
        )

        st.markdown("### Filtered customer metrics")
        f_k1, f_k2, f_k3 = st.columns(3)
        f_k1.metric("Customers in view", f"{filtered_rfm['CustomerID'].nunique():,}")
        f_k2.metric(
            "Revenue in view",
            f"{filtered_rfm['Monetary'].sum():,.2f}"
        )
        f_k3.metric(
            "Average recency (days)",
            f"{filtered_rfm['Recency'].mean():,.1f}"
            if not filtered_rfm.empty else "—"
        )

        st.markdown("### Customer-level RFM table")
        table_columns = [
            col for col in [
                "CustomerID", "Recency", "Frequency", "Monetary",
                "Segment", "SegmentName"
            ] if col in filtered_rfm.columns
        ]
        st.dataframe(
            filtered_rfm[table_columns].round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download filtered customer list (CSV)",
            filtered_rfm.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customer_segments.csv",
            mime="text/csv",
            key="download_filtered_customer_segments",
        )

        st.divider()
        st.markdown("### Segment profile comparison")

        profile = (
            dashboard_rfm.groupby(["Segment", "SegmentName"], dropna=False)
            .agg(
                Customers=("CustomerID", "nunique"),
                AvgRecency=("Recency", "mean"),
                AvgFrequency=("Frequency", "mean"),
                AvgMonetary=("Monetary", "mean"),
                TotalRevenue=("Monetary", "sum"),
            )
            .reset_index()
            .sort_values("Segment")
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Customers per segment**")
            st.bar_chart(
                profile.set_index("SegmentName")[["Customers"]]
            )

            st.markdown("**Total revenue per segment**")
            st.bar_chart(
                profile.set_index("SegmentName")[["TotalRevenue"]]
            )

        with chart_right:
            st.markdown("**Average spending per segment**")
            st.bar_chart(
                profile.set_index("SegmentName")[["AvgMonetary"]]
            )

            st.markdown("**Average purchase frequency**")
            st.bar_chart(
                profile.set_index("SegmentName")[["AvgFrequency"]]
            )

        st.markdown("**Detailed segment profile**")
        st.dataframe(profile.round(2), use_container_width=True, hide_index=True)

        # Segment Details & Action Plan
        st.divider()
        st.markdown("### Segment Details & Action Plan")
        st.write(
            "Choose one segment to inspect its customer profile and view "
            "practical actions to test. These suggestions are guidelines, "
            "not guaranteed outcomes."
        )

        detail_segment = st.selectbox(
            "Choose a segment to explore",
            options=sorted(dashboard_rfm["SegmentName"].unique().tolist()),
            key="segment_details_action_plan",
        )
        detail_profile = profile[profile["SegmentName"] == detail_segment]
        detail_customers = dashboard_rfm[
            dashboard_rfm["SegmentName"] == detail_segment
        ].copy()

        if not detail_profile.empty and not detail_customers.empty:
            segment_row = detail_profile.iloc[0]
            total_customers = max(int(dashboard_rfm["CustomerID"].nunique()), 1)
            customer_share = (
                detail_customers["CustomerID"].nunique() / total_customers * 100
            )
            total_revenue_all = float(dashboard_rfm["Monetary"].sum())
            revenue_share = (
                float(detail_customers["Monetary"].sum()) / total_revenue_all * 100
                if total_revenue_all else 0.0
            )

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Customers", f"{detail_customers['CustomerID'].nunique():,}")
            d2.metric("Share of customers", f"{customer_share:.1f}%")
            d3.metric("Segment revenue", f"{detail_customers['Monetary'].sum():,.2f}")
            d4.metric("Share of revenue", f"{revenue_share:.1f}%")

            p1, p2, p3 = st.columns(3)
            p1.metric("Average spend", f"{detail_customers['Monetary'].mean():,.2f}")
            p2.metric("Average frequency", f"{detail_customers['Frequency'].mean():,.2f}")
            p3.metric(
                "Average recency (days)",
                f"{detail_customers['Recency'].mean():,.1f}",
            )

            avg_recency = float(detail_customers["Recency"].mean())
            avg_frequency = float(detail_customers["Frequency"].mean())
            avg_spend = float(detail_customers["Monetary"].mean())
            overall_frequency = float(dashboard_rfm["Frequency"].mean())
            overall_spend = float(dashboard_rfm["Monetary"].mean())

            if avg_recency >= 60:
                action_heading = "Re-engagement to test"
                action_items = [
                    "Review recent purchase history before contacting customers.",
                    "Test a friendly win-back message with relevant products.",
                    "Measure response and repeat-purchase rates before expanding the campaign.",
                ]
            elif avg_frequency >= overall_frequency and avg_spend >= overall_spend:
                action_heading = "Retention and loyalty to test"
                action_items = [
                    "Consider a loyalty benefit or early access to relevant offers.",
                    "Personalize recommendations using past purchases.",
                    "Track repeat purchase rate and revenue per customer.",
                ]
            elif avg_frequency >= overall_frequency:
                action_heading = "Repeat-purchase growth to test"
                action_items = [
                    "Recommend complementary products based on purchase history.",
                    "Test reminders or bundles that fit customers' buying patterns.",
                    "Compare repeat-purchase rates with a baseline group.",
                ]
            else:
                action_heading = "Customer development to test"
                action_items = [
                    "Test relevant product bundles or introductory offers.",
                    "Use purchase history to avoid irrelevant promotions.",
                    "Monitor average order value and repeat purchases.",
                ]

            st.markdown(f"**Suggested action theme: {action_heading}**")
            for action in action_items:
                st.markdown(f"- {action}")

            st.markdown("**Customers in this segment**")
            detail_columns = [
                col for col in [
                    "CustomerID", "Recency", "Frequency", "Monetary",
                    "Segment", "SegmentName"
                ] if col in detail_customers.columns
            ]
            st.dataframe(
                detail_customers[detail_columns].sort_values(
                    "Monetary", ascending=False
                ).round(2),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected segment customers (CSV)",
                detail_customers.to_csv(index=False).encode("utf-8"),
                file_name="selected_segment_customers.csv",
                mime="text/csv",
                key="download_selected_segment_customers",
            )

        st.markdown("### RFM relationship explorer")
        x_choice = st.selectbox(
            "Horizontal axis",
            options=["Recency", "Frequency", "Monetary"],
            index=0,
            key="rfm_x_axis",
        )
        y_choice = st.selectbox(
            "Vertical axis",
            options=["Monetary", "Frequency", "Recency"],
            index=1,
            key="rfm_y_axis",
        )
        scatter_data = dashboard_rfm[[x_choice, y_choice, "SegmentName"]].copy()
        scatter_data = scatter_data.rename(
            columns={"SegmentName": "Segment label"}
        )
        st.caption(
            "Use these relationships to inspect patterns. Recency is measured "
            "in days; Frequency is the number of orders; Monetary is total spend."
        )
        st.scatter_chart(
            scatter_data,
            x=x_choice,
            y=y_choice,
            color="Segment label",
        )

        st.download_button(
            "Download complete segment profile (CSV)",
            profile.to_csv(index=False).encode("utf-8"),
            file_name="advanced_segment_profile.csv",
            mime="text/csv",
            key="download_advanced_segment_profile",
        )

    # Cluster quality evaluation retained from the existing dashboard.
    st.divider()
    st.subheader("Cluster Quality Evaluation")
    st.write(
        "Evaluate candidate cluster counts using the Elbow Method and "
        "Silhouette Score. These metrics support the choice of K, but "
        "business usefulness should also be considered."
    )

    try:
        evaluation_df = evaluate_clusters(df, max_clusters=8)

        if evaluation_df.empty:
            st.warning("No cluster evaluation results are available.")
        else:
            quality_left, quality_right = st.columns(2)

            with quality_left:
                st.markdown("**Elbow Method — Inertia**")
                st.line_chart(evaluation_df.set_index("K")[["Inertia"]])

            with quality_right:
                st.markdown("**Silhouette Score**")
                st.line_chart(
                    evaluation_df.set_index("K")[["SilhouetteScore"]]
                )

            st.dataframe(
                evaluation_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

            valid_scores = evaluation_df.dropna(subset=["SilhouetteScore"])
            if not valid_scores.empty:
                best_row = valid_scores.loc[
                    valid_scores["SilhouetteScore"].idxmax()
                ]
                st.info(
                    f"Among the tested values, K = {int(best_row['K'])} "
                    f"has the highest silhouette score "
                    f"({float(best_row['SilhouetteScore']):.3f}). "
                    "Compare its customer profiles before changing the "
                    "selected number of segments."
                )
    except (ValueError, KeyError) as exc:
        st.warning(f"Cluster evaluation unavailable: {exc}")


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
            "MAE and RMSE measure test-set error; lower is generally "
            "better. R² can be unstable on small or noisy test sets."
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
            "Forecasts are estimates, not guarantees. Accuracy depends "
            "on data quality, history length, seasonality, and the "
            "forecasting method."
        )

    else:
        st.warning("Forecast results are not available for this dataset.")


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