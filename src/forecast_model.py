import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


FEATURE_DATA_PATH = "data/processed/feature_sales_data.csv"

MODEL_PATH = "models/demand_forecasting_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

METRICS_PATH = "reports/model_metrics.csv"
PREDICTIONS_PATH = "data/forecast_output/model_predictions.csv"


def load_feature_data(input_path: str = FEATURE_DATA_PATH) -> pd.DataFrame:
    """
    Load feature engineered sales dataset.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Feature data file not found: {input_path}")

    df = pd.read_csv(input_path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    return df


def calculate_rmse(y_true, y_pred) -> float:
    """
    Calculate Root Mean Squared Error.
    """

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true, y_pred) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_index = y_true != 0

    if non_zero_index.sum() == 0:
        return 0.0

    return float(
        np.mean(
            np.abs(
                (y_true[non_zero_index] - y_pred[non_zero_index])
                / y_true[non_zero_index]
            )
        )
        * 100
    )


def prepare_train_test_data(df: pd.DataFrame):
    """
    Prepare train and test data using time-based split.
    The latest 20 percent dates are used as test data.
    """

    df = df.copy()
    df = df.sort_values("order_date").reset_index(drop=True)

    target_column = "units_sold"

    # These columns are removed to avoid target leakage.
    leakage_columns = [
        "units_sold",
        "revenue",
        "revenue_per_unit",
    ]

    # order_date is used only for splitting and reporting.
    non_feature_columns = [
        "order_date",
    ]

    drop_columns = leakage_columns + non_feature_columns

    feature_columns = [col for col in df.columns if col not in drop_columns]

    split_date = df["order_date"].quantile(0.80)

    train_df = df[df["order_date"] <= split_date].copy()
    test_df = df[df["order_date"] > split_date].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    print("Train-test split completed.")
    print(f"Split date: {split_date.date()}")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(f"Feature count: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, test_df, feature_columns


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for categorical and numeric columns.

    Categorical columns are defined explicitly to avoid pandas dtype warnings
    and to keep preprocessing stable across pandas versions.
    """

    expected_categorical_columns = [
        "store_id",
        "product_id",
        "category",
        "region",
    ]

    categorical_columns = [
        col for col in expected_categorical_columns if col in X_train.columns
    ]

    numeric_columns = [
        col for col in X_train.columns if col not in categorical_columns
    ]

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumeric columns:")
    print(numeric_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns,
            ),
        ]
    )

    return preprocessor


def train_random_forest_model(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Train RandomForestRegressor as a baseline forecasting model.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def train_xgboost_model(preprocessor: ColumnTransformer) -> Pipeline:
    """
    Train XGBoost regression model for demand forecasting.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def evaluate_model(model_name: str, model: Pipeline, X_test, y_test) -> dict:
    """
    Evaluate trained model using common regression metrics.
    """

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = calculate_rmse(y_test, predictions)
    mape = calculate_mape(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "model_name": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
        "R2": round(float(r2), 4),
    }

    return metrics


def save_outputs(
    best_model: Pipeline,
    feature_columns: list,
    metrics_df: pd.DataFrame,
    best_predictions: np.ndarray,
    test_df: pd.DataFrame,
) -> None:
    """
    Save model, feature columns, metrics and prediction output.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/forecast_output", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    metrics_df.to_csv(METRICS_PATH, index=False)

    prediction_output = test_df[
        [
            "order_date",
            "store_id",
            "product_id",
            "category",
            "region",
            "units_sold",
        ]
    ].copy()

    prediction_output["predicted_units_sold"] = np.round(
        np.maximum(best_predictions, 0)
    ).astype(int)

    prediction_output["forecast_error"] = (
        prediction_output["units_sold"]
        - prediction_output["predicted_units_sold"]
    )

    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    print(f"Model metrics saved to: {METRICS_PATH}")
    print(f"Prediction output saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    print("Loading feature engineered data...")
    sales_df = load_feature_data()

    X_train, X_test, y_train, y_test, test_df, feature_columns = prepare_train_test_data(
        sales_df
    )

    preprocessor = build_preprocessor(X_train)

    print("\nTraining Random Forest baseline model...")
    random_forest_model = train_random_forest_model(preprocessor)
    random_forest_model.fit(X_train, y_train)

    print("Evaluating Random Forest model...")
    rf_metrics = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test,
    )

    print("\nTraining XGBoost forecasting model...")
    xgboost_model = train_xgboost_model(preprocessor)
    xgboost_model.fit(X_train, y_train)

    print("Evaluating XGBoost model...")
    xgb_metrics = evaluate_model(
        "XGBoost",
        xgboost_model,
        X_test,
        y_test,
    )

    metrics_df = pd.DataFrame([rf_metrics, xgb_metrics])

    print("\nModel Evaluation Results:")
    print(metrics_df)

    # Select best model based on lowest RMSE.
    best_model_name = metrics_df.sort_values("RMSE").iloc[0]["model_name"]

    if best_model_name == "XGBoost":
        best_model = xgboost_model
    else:
        best_model = random_forest_model

    print(f"\nBest model selected: {best_model_name}")

    best_predictions = best_model.predict(X_test)

    save_outputs(
        best_model=best_model,
        feature_columns=feature_columns,
        metrics_df=metrics_df,
        best_predictions=best_predictions,
        test_df=test_df,
    )

    print("\nForecast model training completed successfully.")