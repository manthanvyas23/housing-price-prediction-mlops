import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_pipeline.inference import predict


def create_sample_df():
    """Create a tiny engineered dataset for inference."""

    return pd.DataFrame(
        {
            "year": [2022, 2023],
            "quarter": [1, 2],
            "month": [1, 6],
            "median_list_price": [300000, 350000],
            "median_ppsf": [250, 275],
            "median_list_ppsf": [260, 285],
            "homes_sold": [50, 60],
            "pending_sales": [55, 62],
            "new_listings": [40, 45],
            "inventory": [120, 130],
            "median_dom": [30, 28],
            "avg_sale_to_list": [1.01, 1.02],
            "sold_above_list": [0.40, 0.45],
            "off_market_in_two_weeks": [0.25, 0.30],
            "bank": [2, 1],
            "bus": [5, 4],
            "hospital": [1, 1],
            "mall": [2, 3],
            "park": [6, 7],
            "restaurant": [15, 18],
            "school": [8, 9],
            "station": [3, 2],
            "supermarket": [4, 5],
            "Total Population": [100000, 120000],
            "Median Age": [35, 36],
            "Per Capita Income": [55000, 57000],
            "Total Families Below Poverty": [4000, 4200],
            "Total Housing Units": [50000, 52000],
            "Median Rent": [1800, 1850],
            "Median Home Value": [450000, 470000],
            "Total Labor Force": [60000, 62000],
            "Unemployed Population": [2500, 2600],
            "Total School Age Population": [18000, 18500],
            "Total School Enrollment": [17000, 17500],
            "Median Commute Time": [28, 30],
            "lat": [37.77, 37.80],
            "lng": [-122.42, -122.40],
            "zipcode_freq": [15, 20],
            "city_full_encoded": [420000, 430000],
            "price": [460000, 480000],
        }
    )


def test_inference_runs_and_returns_predictions():
    sample_df = create_sample_df()

    preds_df = predict(sample_df)

    assert not preds_df.empty
    assert "predicted_price" in preds_df.columns
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])

    print(preds_df.head())
