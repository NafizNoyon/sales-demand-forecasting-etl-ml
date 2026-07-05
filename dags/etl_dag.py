"""
Airflow DAG for Sales Demand Forecasting ETL and ML Pipeline.

This DAG demonstrates an automated workflow for:
1. Raw sales data ingestion
2. Data cleaning
3. Data quality validation
4. Feature engineering
5. Forecast model training
6. Model evaluation report generation

Note:
This DAG is designed for a Linux-based Airflow environment.
For local Windows development, run the project scripts manually or use a local runner.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.data_ingestion import create_demo_sales_data, load_raw_sales_data
from src.data_cleaning import load_raw_data, clean_sales_data, save_clean_data
from src.data_validation import (
    load_cleaned_data,
    validate_sales_data,
    create_data_profile,
    save_validation_summary,
    generate_html_report,
)
from src.feature_engineering import (
    load_cleaned_data as load_cleaned_data_for_features,
    create_forecasting_features,
    save_feature_data,
)


default_args = {
    "owner": "data-engineering-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def ingest_raw_sales_data():
    """
    Create and load raw sales data.
    """

    create_demo_sales_data()
    df = load_raw_sales_data()

    print("Raw sales data ingestion completed.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


def clean_raw_sales_data():
    """
    Clean raw sales data and save processed output.
    """

    raw_df = load_raw_data()
    cleaned_df = clean_sales_data(raw_df)
    save_clean_data(cleaned_df)

    print("Sales data cleaning completed.")
    print(f"Cleaned rows: {cleaned_df.shape[0]}")


def validate_cleaned_sales_data():
    """
    Run data quality validation checks and generate report.
    """

    sales_df = load_cleaned_data()

    validation_results = validate_sales_data(sales_df)
    profile_summary = create_data_profile(sales_df)

    save_validation_summary(validation_results)
    generate_html_report(validation_results, profile_summary)

    failed_checks = validation_results[validation_results["status"] == "FAIL"]

    if not failed_checks.empty:
        raise ValueError(f"Data quality validation failed: {failed_checks}")

    print("Data quality validation completed successfully.")
    print("All validation checks passed.")


def engineer_forecasting_features():
    """
    Create forecasting features and save ML-ready dataset.
    """

    cleaned_df = load_cleaned_data_for_features()
    feature_df = create_forecasting_features(cleaned_df)
    save_feature_data(feature_df)

    print("Feature engineering completed.")
    print(f"Feature dataset rows: {feature_df.shape[0]}")
    print(f"Feature dataset columns: {feature_df.shape[1]}")


def train_forecasting_model():
    """
    Run forecast model training script.
    """

    exit_code = os.system(f"{sys.executable} src/forecast_model.py")

    if exit_code != 0:
        raise RuntimeError("Forecast model training failed.")

    print("Forecast model training completed successfully.")


def evaluate_forecasting_model():
    """
    Run model evaluation script and generate reports.
    """

    exit_code = os.system(f"{sys.executable} src/model_evaluation.py")

    if exit_code != 0:
        raise RuntimeError("Model evaluation failed.")

    print("Model evaluation completed successfully.")


with DAG(
    dag_id="sales_demand_forecasting_etl_ml_pipeline",
    default_args=default_args,
    description="Automated ETL, data validation, feature engineering and ML forecasting pipeline.",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["etl", "data-quality", "machine-learning", "forecasting"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_raw_sales_data",
        python_callable=ingest_raw_sales_data,
    )

    clean_task = PythonOperator(
        task_id="clean_raw_sales_data",
        python_callable=clean_raw_sales_data,
    )

    validate_task = PythonOperator(
        task_id="validate_cleaned_sales_data",
        python_callable=validate_cleaned_sales_data,
    )

    feature_task = PythonOperator(
        task_id="engineer_forecasting_features",
        python_callable=engineer_forecasting_features,
    )

    train_model_task = PythonOperator(
        task_id="train_forecasting_model",
        python_callable=train_forecasting_model,
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_forecasting_model",
        python_callable=evaluate_forecasting_model,
    )

    (
        ingest_task
        >> clean_task
        >> validate_task
        >> feature_task
        >> train_model_task
        >> evaluate_model_task
    )