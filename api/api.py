import os
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = "models/demand_forecasting_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
FEATURE_DATA_PATH = "data/processed/feature_sales_data.csv"


app = FastAPI(
    title="Sales Demand Forecasting API",
    description="API for predicting sales demand using a trained machine learning model.",
    version="1.0.0",
)


class ForecastInput(BaseModel):
    store_id: str
    product_id: str
    category: str
    region: str

    unit_price: float
    discount: float
    marketing_spend: float
    holiday: int

    year: int
    month: int
    day: int
    day_of_week: int
    week_of_year: int
    quarter: int

    is_weekend: int
    is_month_start: int
    is_month_end: int

    units_sold_lag_1: float
    revenue_lag_1: float
    units_sold_lag_7: float
    revenue_lag_7: float
    units_sold_lag_14: float
    revenue_lag_14: float
    units_sold_lag_28: float
    revenue_lag_28: float

    units_sold_rolling_mean_7: float
    units_sold_rolling_std_7: float
    revenue_rolling_mean_7: float

    units_sold_rolling_mean_14: float
    units_sold_rolling_std_14: float
    revenue_rolling_mean_14: float

    units_sold_rolling_mean_28: float
    units_sold_rolling_std_28: float
    revenue_rolling_mean_28: float

    price_after_discount: float
    marketing_spend_per_unit: float


def load_model_and_features():
    """
    Load trained forecasting model and feature columns.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, feature_columns


def prepare_input_data(input_data: Dict[str, Any], feature_columns: list) -> pd.DataFrame:
    """
    Convert API input into model-ready DataFrame.
    """

    input_df = pd.DataFrame([input_data])

    missing_columns = [col for col in feature_columns if col not in input_df.columns]

    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    input_df = input_df[feature_columns]

    return input_df


@app.get("/")
def home():
    """
    Home endpoint.
    """

    return {
        "message": "Sales Demand Forecasting API is running.",
        "project": "Sales Demand Forecasting with Automated ETL and Data Quality Checks",
        "docs": "Open /docs to test the API.",
    }


@app.get("/health")
def health_check():
    """
    Check whether model files are available.
    """

    model_exists = os.path.exists(MODEL_PATH)
    feature_columns_exists = os.path.exists(FEATURE_COLUMNS_PATH)

    status = "healthy" if model_exists and feature_columns_exists else "unhealthy"

    return {
        "status": status,
        "model_exists": model_exists,
        "feature_columns_exists": feature_columns_exists,
    }


@app.post("/predict")
def predict_demand(input_data: ForecastInput):
    """
    Predict sales demand from manual input.
    """

    try:
        model, feature_columns = load_model_and_features()

        input_dict = input_data.model_dump()
        input_df = prepare_input_data(input_dict, feature_columns)

        prediction = model.predict(input_df)[0]
        prediction = max(float(prediction), 0)

        return {
            "predicted_units_sold": round(prediction, 2),
            "predicted_units_sold_rounded": int(round(prediction)),
            "message": "Demand prediction completed successfully.",
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/sample-predict")
def sample_predict():
    """
    Run prediction using one sample row from feature data.
    This endpoint is useful for quickly testing the deployed API.
    """

    try:
        if not os.path.exists(FEATURE_DATA_PATH):
            raise FileNotFoundError(f"Feature data file not found: {FEATURE_DATA_PATH}")

        model, feature_columns = load_model_and_features()

        df = pd.read_csv(FEATURE_DATA_PATH)

        sample_row = df.iloc[0]
        input_df = pd.DataFrame([sample_row[feature_columns]])

        prediction = model.predict(input_df)[0]
        prediction = max(float(prediction), 0)

        actual_units_sold = int(sample_row["units_sold"])

        return {
            "sample_store_id": sample_row["store_id"],
            "sample_product_id": sample_row["product_id"],
            "sample_category": sample_row["category"],
            "actual_units_sold": actual_units_sold,
            "predicted_units_sold": round(prediction, 2),
            "predicted_units_sold_rounded": int(round(prediction)),
            "message": "Sample prediction completed successfully.",
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))