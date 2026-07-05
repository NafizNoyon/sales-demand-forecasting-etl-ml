import subprocess
import sys
import time
from datetime import datetime


PIPELINE_STEPS = [
    {"name": "Data Ingestion", "command": ["src/data_ingestion.py"]},
    {"name": "Data Cleaning", "command": ["src/data_cleaning.py"]},
    {"name": "Data Quality Validation", "command": ["src/data_validation.py"]},
    {"name": "Feature Engineering", "command": ["src/feature_engineering.py"]},
    {"name": "Forecast Model Training", "command": ["src/forecast_model.py"]},
    {"name": "Model Evaluation Report", "command": ["src/model_evaluation.py"]},
]


def run_step(step_name: str, command: list) -> None:
    print("=" * 80)
    print(f"Starting step: {step_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable] + command,
        text=True,
    )

    duration = round(time.time() - start_time, 2)

    if result.returncode != 0:
        print("\nPipeline failed.")
        print(f"Failed step: {step_name}")
        print(f"Duration before failure: {duration} seconds")
        sys.exit(result.returncode)

    print("\nStep completed successfully.")
    print(f"Completed step: {step_name}")
    print(f"Duration: {duration} seconds")
    print()


def main() -> None:
    print("\nSales Demand Forecasting Pipeline Started")
    print(f"Pipeline start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    pipeline_start_time = time.time()

    for step in PIPELINE_STEPS:
        run_step(step["name"], step["command"])

    total_duration = round(time.time() - pipeline_start_time, 2)

    print("=" * 80)
    print("Pipeline completed successfully.")
    print(f"Total duration: {total_duration} seconds")
    print("=" * 80)

    print("\nGenerated Outputs:")
    print("- Raw data: data/raw/sales_data.csv")
    print("- Cleaned data: data/processed/cleaned_sales_data.csv")
    print("- Feature data: data/processed/feature_sales_data.csv")
    print("- Data quality report: reports/data_quality_report.html")
    print("- Model file: models/demand_forecasting_model.pkl")
    print("- Model metrics: reports/model_metrics.csv")
    print("- Model evaluation report: reports/model_evaluation_report.html")
    print("- Forecast predictions: data/forecast_output/model_predictions.csv")


if __name__ == "__main__":
    main()
