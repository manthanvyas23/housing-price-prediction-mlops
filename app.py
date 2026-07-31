import logging
import os
import time
from pathlib import Path

import boto3
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from joblib import load

# ==========================================
# Configure Logging
# ==========================================
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Housing Price Prediction Dashboard started.")

st.set_page_config(
    page_title="Housing Price Prediction Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# Session State
# ==========================================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==========================================
# Monitoring
# ==========================================
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if "dashboard_start_time" not in st.session_state:
    st.session_state.dashboard_start_time = pd.Timestamp.now()

if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = None

if "records_processed" not in st.session_state:
    st.session_state.records_processed = 0

if "average_records" not in st.session_state:
    st.session_state.average_records = 0

# ============================
# Sidebar
# ============================

st.sidebar.title("🏠 Housing Dashboard")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ Tech Stack")

st.sidebar.markdown("""
- FastAPI
- Streamlit
- XGBoost
- AWS S3
- Railway
- MLflow
- GitHub Actions
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Dashboard Monitoring")

uptime = pd.Timestamp.now() - st.session_state.dashboard_start_time

st.sidebar.metric(
    "📊 Total Predictions",
    st.session_state.prediction_count,
)

st.sidebar.metric(
    "📦 Records Processed",
    f"{st.session_state.records_processed:,}",
)

st.sidebar.metric(
    "📈 Average Records per Request",
    f"{st.session_state.average_records:,.0f}",
)

if st.session_state.last_prediction_time is None:
    last_prediction_time = "N/A"
else:
    last_prediction_time = f"{st.session_state.last_prediction_time:.3f} sec"

st.sidebar.metric(
    "⚡️ Prediction Latency",
    last_prediction_time,
)

total_seconds = int(uptime.total_seconds())

hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60

formatted_uptime = f"{hours:02}:{minutes:02}:{seconds:02}"

st.sidebar.metric(
    "⏱️ Dashboard Uptime",
    formatted_uptime,
)

st.sidebar.success("🟢 Status: Running")

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Model Information")

st.sidebar.write("**Model:** XGBoost Regressor")
st.sidebar.write("**Expected Features:** 39")
st.sidebar.write("**Model File:** xgb_best_model.pkl")
st.sidebar.caption("Version 1.0.0")

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "housing-price-prediction-mlops")
REGION = os.getenv("AWS_REGION", "ap-southeast-2")

s3 = boto3.client("s3", region_name=REGION)


def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        st.info(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)


# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv",
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/cleaning_holdout.csv", "data/processed/cleaning_holdout.csv"
)
MODEL_PATH = load_from_s3(
    "models/xgb_best_model.pkl",
    "models/xgb_best_model.pkl",
)


@st.cache_resource
def load_model():
    return load(MODEL_PATH)


model = load_model()


# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp


fe_df, disp_df = load_data()

# ============================
# UI
# ============================
st.title("🏠 Housing Price Prediction Dashboard")

st.markdown("""
Analyze machine learning predictions on unseen housing data.
Compare predicted prices against actual prices using interactive filters,
evaluation metrics, and visualizations.
""")

st.divider()

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

st.subheader("🔎 Filters")

with st.container():

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        year = st.selectbox("📅 Year", years, index=0)
    with col2:
        month = st.selectbox("📅 Month", months, index=0)
    with col3:
        region = st.selectbox("🌍 Region", regions, index=0)

if st.button("🚀 Generate Predictions"):
    start_time = time.perf_counter()
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= disp_df["region"] == region

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.success(f"Showing predictions for {year}-{month:02d} | Region: {region}")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            with st.spinner("🔄 Generating predictions... Please wait..."):
                resp = requests.post(API_URL, json=payload, timeout=60)  # type: ignore
                resp.raise_for_status()
                out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(
                    float
                )

            logger.info(
                f"Prediction generated | "
                f"Region={region} | "
                f"Year={year} | "
                f"Month={month} | "
                f"Records={len(view)} | "
                f"Average Prediction=${view['prediction'].mean():,.2f}"
            )

            # ==========================================
            # Monitoring
            # ==========================================
            st.session_state.prediction_count += 1

            records = len(view)

            st.session_state.records_processed += records

            st.session_state.average_records = (
                st.session_state.records_processed / st.session_state.prediction_count
            )

            elapsed_time = time.perf_counter() - start_time
            st.session_state.last_prediction_time = elapsed_time

            # ==========================================
            # Save prediction run summary
            # ==========================================
            history_entry = {
                "Run Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Year": year,
                "Month": month,
                "Region": region,
                "Records": len(view),
                "Average Prediction": round(view["prediction"].mean(), 2),
                "Minimum Prediction": round(view["prediction"].min(), 2),
                "Maximum Prediction": round(view["prediction"].max(), 2),
                "Prediction Time (s)": round(elapsed_time, 3),
            }

            st.session_state.prediction_history.append(history_entry)

            # Keep only the last 10 runs
            st.session_state.prediction_history = st.session_state.prediction_history[
                -10:
            ]

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = (
                (view["prediction"] - view["actual_price"]).abs() / view["actual_price"]
            ).mean() * 100

            display_view = view.copy()

            display_view["date"] = pd.to_datetime(display_view["date"]).dt.strftime(
                "%d %b %Y"
            )

            display_view["actual_price"] = (
                display_view["actual_price"].round(0).map(lambda x: f"${x:,.0f}")
            )

            display_view["prediction"] = (
                display_view["prediction"].round(0).map(lambda x: f"${x:,.0f}")
            )

            st.subheader("📋 Prediction Results")

            results_df = display_view[
                ["date", "region", "actual_price", "prediction"]
            ].reset_index(drop=True)

            st.dataframe(results_df, width="stretch")

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name=f"housing_predictions_{year}_{month:02d}.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("📉 MAE", f"${mae:,.0f}")
            with c2:
                st.metric("📊 RMSE", f"${rmse:,.0f}")
            with c3:
                st.metric("🎯 Avg Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)  # type: ignore
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(
                    preds_all, index=yearly_data.index
                ).astype(float)

            else:
                yearly_data = disp_df[
                    (disp_df["year"] == year) & (disp_df["region"] == region)
                ].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)  # type: ignore
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(
                    preds_region, index=yearly_data.index
                ).astype(float)

            # Aggregate by month
            monthly_avg = (
                yearly_data.groupby("month")[["actual_price", "prediction"]]
                .mean()
                .reset_index()
            )

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(
                lambda m: "Selected" if m == month else "Other"
            )

            # ============================
            # Prediction Distribution
            # ============================
            st.markdown("---")
            st.subheader("📊 Prediction Distribution")

            distribution_df = view[["actual_price", "prediction"]].copy()

            distribution_df = distribution_df.melt(var_name="Type", value_name="Price")

            fig_distribution = px.histogram(
                distribution_df,
                x="Price",
                color="Type",
                barmode="overlay",
                opacity=0.7,
                nbins=30,
                labels={"Price": "House Price ($)", "Type": ""},
            )

            fig_distribution.update_layout(
                template="plotly_white",
                height=450,
                legend_title_text="",
            )

            fig_distribution.for_each_trace(
                lambda t: t.update(
                    name=(
                        "Actual Price"
                        if t.name == "actual_price"
                        else "Predicted Price"
                    )
                )
            )

            st.plotly_chart(fig_distribution, width="stretch")

            # ============================
            # Region Comparison
            # ============================
            st.markdown("---")
            st.subheader("🌍 Region Comparison")

            region_avg = (
                view.groupby("region")[["actual_price", "prediction"]]
                .mean()
                .reset_index()
            )

            region_avg = region_avg.sort_values("actual_price", ascending=False).head(
                10
            )

            region_chart = region_avg.melt(
                id_vars="region",
                value_vars=["actual_price", "prediction"],
                var_name="Type",
                value_name="Average Price",
            )

            fig_region = px.bar(
                region_chart,
                x="Average Price",
                y="region",
                color="Type",
                orientation="h",
                barmode="group",
                labels={
                    "region": "Region",
                    "Average Price": "Average House Price ($)",
                    "Type": "",
                },
            )

            fig_region.update_layout(
                template="plotly_white",
                height=500,
                legend_title_text="",
            )

            fig_region.for_each_trace(
                lambda t: t.update(
                    name=(
                        "Actual Price"
                        if t.name == "actual_price"
                        else "Predicted Price"
                    )
                )
            )

            st.plotly_chart(fig_region, width="stretch")

            # ============================
            # Feature Importance
            # ============================
            st.markdown("---")
            st.subheader("📌 Feature Importance")

            feature_names = fe_df.columns[:-1]

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": model.feature_importances_,
                }
            )

            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            ).head(15)

            fig_importance = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=None,
            )

            fig_importance.update_layout(
                template="plotly_white",
                height=550,
            )

            fig_importance.update_yaxes(autorange="reversed")

            st.plotly_chart(fig_importance, width="stretch")

            st.markdown("---")
            st.subheader("📈 Yearly Prediction Trend")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=None,
            )

            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                height=500,
                legend_title_text="",
                xaxis_title="Month",
                yaxis_title="House Price ($)",
                title={"text": ""},
                font={"size": 14},
            )

            fig.update_traces(line={"width": 3}, marker={"size": 8})

            fig.for_each_trace(
                lambda t: t.update(
                    name=(
                        "Actual Price"
                        if t.name == "actual_price"
                        else "Predicted Price"
                    )
                )
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=month - 0.5,
                x1=month + 0.5,
                fillcolor="gold",
                opacity=0.15,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, width="stretch")

            # ==========================================
            # Prediction History
            # ==========================================
            st.markdown("---")
            st.subheader("📝 Prediction History")

            col1, col2 = st.columns([6, 1])

            with col2:
                if st.button("🗑️ Clear History"):

                    st.session_state.prediction_history = []

                    st.session_state.prediction_count = 0
                    st.session_state.records_processed = 0
                    st.session_state.average_records = 0
                    st.session_state.last_prediction_time = None

                    st.rerun()

            if st.session_state.prediction_history:
                history = pd.DataFrame(st.session_state.prediction_history)

                # Show newest prediction first
                history = history.iloc[::-1].reset_index(drop=True)

                for col in [
                    "Average Prediction",
                    "Minimum Prediction",
                    "Maximum Prediction",
                ]:
                    history[col] = history[col].map(lambda x: f"${x:,.2f}")

                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No predictions have been made yet.")

        except requests.exceptions.ConnectionError:
            logger.error("Connection to Prediction API failed.")

            st.error("❌ Unable to connect to the Prediction API.")
            st.info("Please make sure the FastAPI server is running on port 8000.")

        except requests.exceptions.Timeout:
            logger.error("Prediction API request timed out.")

            st.error("⏱️ The Prediction API took too long to respond.")

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"
            logger.error(f"Prediction API returned HTTP {status_code}")
            st.error(f"⚠️ API returned an error: {status_code}")

        except requests.exceptions.RequestException:
            logger.exception("Unexpected network error occurred.")
            st.error(
                "❌ An unexpected network error occurred while generating predictions."
            )

else:
    st.caption(
        "👆 Select a year, month, and region, then click **🚀 Generate Predictions**."
    )
