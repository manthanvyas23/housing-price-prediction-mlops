import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import boto3, os
from pathlib import Path

st.set_page_config(
    page_title = "Housing Price Prediction Dashboard",
    page_icon = "🏡",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# ============================
# Sidebar
# ============================

st.sidebar.title("🏠 Housing Dashboard")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### 📌 Project

**Model**
- XGBoost Regressor

**Dataset**
- Zillow Housing Prices

**Framework**
- Streamlit + FastAPI

**Cloud Storage**
- AWS S3
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ System Status")

try:
    health = requests.get("http://127.0.0.1:8000/health", timeout = 5)
    health.raise_for_status()

    health_data = health.json()

    st.sidebar.success("✅ API Connected")

    if health_data["status"] == "healthy":
        st.sidebar.success("✅ Model Loaded")

        st.sidebar.markdown("### 📊 Model Details")

        st.sidebar.write(
            f"*Expected Features:* {health_data.get('n_features_expected', 'Unknown')}"
        )

        st.sidebar.write(
            f"*Model File:* {Path(health_data['model_path']).name}"
        )

    else:
        st.sidebar.error("❌ Model Not Loaded")

except Exception:
    st.sidebar.error("❌ API Offline")

st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
S3_BUCKET = os.getenv("S3_BUCKET", "housing-price-prediction-mlops")
REGION = os.getenv("AWS_REGION", "ap-southeast-2")

s3 = boto3.client("s3", region_name = REGION)

def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok = True)
        st.info(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Paths (ensure available locally by fetching from S3 if missing)
HOLDOUT_ENGINEERED_PATH = load_from_s3(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_s3(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates = ["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index = fe.index)
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

st.markdown(
"""
Analyze machine learning predictions on unseen housing data.
Compare predicted prices against actual prices using interactive filters,
evaluation metrics, and visualizations.
"""
)

st.divider()

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

st.subheader("🔎 Filters")

with st.container():
    
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        year = st.selectbox("📅 Year", years, index = 0)
    with col2:
        month = st.selectbox("📅 Month", months, index = 0)
    with col3:
        region = st.selectbox("🌍 Region", regions, index = 0)

if st.button("🚀 Generate Predictions"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.success(f"Showing predictions for {year}-{month:02d} | Region: {region}")

        payload = fe_df.loc[idx].to_dict(orient = "records")

        try:
            with st.spinner("🔄 Generating predictions... Please wait..."):
                resp = requests.post(API_URL, json = payload, timeout = 60) # type: ignore
                resp.raise_for_status()
                out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index = view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index = view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            display_view = view.copy()

            display_view["date"] = pd.to_datetime(
                display_view["date"]
            ).dt.strftime("%d %b %Y")
            
            display_view["actual_price"] = (
                 display_view["actual_price"]
                 .round(0)
                 .map(lambda x: f"${x:,.0f}")
            )
            
            display_view["prediction"] = (
                display_view["prediction"]
                .round(0)
                .map(lambda x: f"${x:,.0f}")
            )

            st.subheader("📋 Prediction Results")

            results_df = display_view[["date", "region", "actual_price", "prediction"]].reset_index(drop = True)

            st.dataframe(
                 results_df,
                 width = "stretch"
            )

            csv = results_df.to_csv(index = False).encode("utf-8")

            st.download_button(
                 label = "📥 Download Predictions (CSV)",
                 data = csv,
                 file_name = f"housing_predictions_{year}_{month:02d}.csv",
                 mime = "text/csv"
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
                payload_all = fe_df.loc[idx_all].to_dict(orient = "records")

                resp_all = requests.post(API_URL, json = payload_all, timeout = 60) # type: ignore
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index = yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient = "records")

                resp_region = requests.post(API_URL, json = payload_region, timeout = 60) # type: ignore
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index = yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == month else "Other")

            st.markdown("---")
            st.subheader("📈 Yearly Prediction Trend")

            fig = px.line(
                monthly_avg,
                x = "month",
                y = ["actual_price", "prediction"],
                markers = True,
                labels = {"value": "Price", "month": "Month"},
                title = None
            )

            fig.update_layout(
                template = "plotly_white",
                hovermode = "x unified",
                height = 500,
                legend_title_text = "",
                xaxis_title = "Month",
                yaxis_title = "House Price ($)",
                title = dict(text = ""),
                font = dict(size = 14)
            )

            fig.update_traces(
                line = dict(width = 3),
                marker = dict(size = 8)
            )

            fig.for_each_trace(
                lambda t: t.update(
                    name = "Actual Price" if t.name == "actual_price" else "Predicted Price"
                )
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0 = month - 0.5,
                x1 = month + 0.5,
                fillcolor = "gold",
                opacity = 0.15,
                layer = "below",
                line_width = 0
            )

            st.plotly_chart(fig, width = "stretch")

        except requests.exceptions.ConnectionError:
            st.error("❌ Unable to connect to the Prediction API.")
            st.info("Please make sure the FastAPI server is running on port 8000.")

        except requests.exceptions.Timeout:
            st.error("⏱️ The Prediction API took too long to respond.")

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"
            st.error(f"⚠️ API returned an error: {status_code}")

        except Exception:
            st.error("❌ An unexpected error occurred while generating predictions.")

else:
    st.caption("👆 Select a year, month, and region, then click **🚀 Generate Predictions**.")