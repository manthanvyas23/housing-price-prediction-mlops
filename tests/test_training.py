from pathlib import Path

import numpy as np
import pandas as pd

from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.train import train_model
from src.training_pipeline.tune import tune_model


def create_dummy_dataset(n_rows=50):
    """Create a synthetic housing dataset matching the engineered schema."""

    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "year": rng.integers(2020, 2025, n_rows),
            "quarter": rng.integers(1, 5, n_rows),
            "month": rng.integers(1, 13, n_rows),
            "median_list_price": rng.uniform(200000, 800000, n_rows),
            "median_ppsf": rng.uniform(100, 500, n_rows),
            "median_list_ppsf": rng.uniform(100, 500, n_rows),
            "homes_sold": rng.integers(10, 500, n_rows),
            "pending_sales": rng.integers(10, 500, n_rows),
            "new_listings": rng.integers(10, 500, n_rows),
            "inventory": rng.integers(50, 500, n_rows),
            "median_dom": rng.uniform(10, 120, n_rows),
            "avg_sale_to_list": rng.uniform(0.8, 1.2, n_rows),
            "sold_above_list": rng.uniform(0, 1, n_rows),
            "off_market_in_two_weeks": rng.uniform(0, 1, n_rows),
            "bank": rng.integers(0, 5, n_rows),
            "bus": rng.integers(0, 10, n_rows),
            "hospital": rng.integers(0, 5, n_rows),
            "mall": rng.integers(0, 5, n_rows),
            "park": rng.integers(0, 10, n_rows),
            "restaurant": rng.integers(0, 20, n_rows),
            "school": rng.integers(0, 20, n_rows),
            "station": rng.integers(0, 10, n_rows),
            "supermarket": rng.integers(0, 10, n_rows),
            "Total Population": rng.integers(1000, 100000, n_rows),
            "Median Age": rng.uniform(20, 60, n_rows),
            "Per Capita Income": rng.uniform(20000, 100000, n_rows),
            "Total Families Below Poverty": rng.integers(100, 5000, n_rows),
            "Total Housing Units": rng.integers(1000, 50000, n_rows),
            "Median Rent": rng.uniform(800, 3500, n_rows),
            "Median Home Value": rng.uniform(100000, 900000, n_rows),
            "Total Labor Force": rng.integers(1000, 50000, n_rows),
            "Unemployed Population": rng.integers(100, 5000, n_rows),
            "Total School Age Population": rng.integers(1000, 50000, n_rows),
            "Total School Enrollment": rng.integers(1000, 50000, n_rows),
            "Median Commute Time": rng.uniform(10, 90, n_rows),
            "lat": rng.uniform(35, 45, n_rows),
            "lng": rng.uniform(-120, -70, n_rows),
            "zipcode_freq": rng.integers(1, 500, n_rows),
            "city_full_encoded": rng.uniform(100000, 900000, n_rows),
            "price": rng.uniform(200000, 900000, n_rows),
        }
    )

    return df


def write_train_eval_csvs(tmp_path):
    train_df = create_dummy_dataset(100)
    eval_df = create_dummy_dataset(40)

    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_model(tmp_path):
    train_path, eval_path = write_train_eval_csvs(tmp_path)

    model_path = tmp_path / "xgb_model.pkl"

    model, metrics = train_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_path,
        model_params={
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    assert model is not None
    assert model_path.exists()

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_evaluate_model(tmp_path):
    train_path, eval_path = write_train_eval_csvs(tmp_path)

    model_path = tmp_path / "xgb_model.pkl"

    train_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_path,
        model_params={
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    metrics = evaluate_model(
        model_path=model_path,
        eval_path=eval_path,
    )

    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_tune_model(tmp_path):
    train_path, eval_path = write_train_eval_csvs(tmp_path)

    model_path = tmp_path / "best_model.pkl"

    best_params, best_metrics = tune_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_path,
        n_trials=1,
        tracking_uri=f"file://{tmp_path}/mlruns",
    )

    assert model_path.exists()

    assert isinstance(best_params, dict)
    assert isinstance(best_metrics, dict)

    assert "rmse" in best_metrics
    assert "mae" in best_metrics
    assert "r2" in best_metrics