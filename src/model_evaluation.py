import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


METRICS_PATH = "reports/model_metrics.csv"
PREDICTIONS_PATH = "data/forecast_output/model_predictions.csv"

REPORT_PATH = "reports/model_evaluation_report.html"
METRICS_CHART_PATH = "reports/model_metrics_comparison.png"
ACTUAL_VS_PREDICTED_PATH = "reports/actual_vs_predicted_demand.png"
ERROR_DISTRIBUTION_PATH = "reports/forecast_error_distribution.png"
SCATTER_PLOT_PATH = "reports/predicted_vs_actual_scatter.png"


def load_model_outputs():
    """
    Load model metrics and prediction output.
    """

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Model metrics file not found: {METRICS_PATH}")

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Prediction output file not found: {PREDICTIONS_PATH}")

    metrics_df = pd.read_csv(METRICS_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    predictions_df["order_date"] = pd.to_datetime(
        predictions_df["order_date"],
        errors="coerce",
    )

    return metrics_df, predictions_df


def create_metrics_comparison_chart(metrics_df: pd.DataFrame) -> None:
    """
    Create model metrics comparison chart.
    """

    os.makedirs("reports", exist_ok=True)

    metric_columns = ["MAE", "RMSE", "MAPE"]

    chart_df = metrics_df.set_index("model_name")[metric_columns]

    ax = chart_df.plot(kind="bar", figsize=(10, 6))

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.legend(title="Metrics")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(METRICS_CHART_PATH, dpi=300)
    plt.close()

    print(f"Metrics comparison chart saved to: {METRICS_CHART_PATH}")


def create_actual_vs_predicted_chart(predictions_df: pd.DataFrame) -> None:
    """
    Create actual vs predicted demand line chart by date.
    """

    daily_forecast = (
        predictions_df.groupby("order_date")[["units_sold", "predicted_units_sold"]]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_forecast["order_date"],
        daily_forecast["units_sold"],
        label="Actual Demand",
    )

    plt.plot(
        daily_forecast["order_date"],
        daily_forecast["predicted_units_sold"],
        label="Predicted Demand",
    )

    plt.title("Actual vs Predicted Sales Demand")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(ACTUAL_VS_PREDICTED_PATH, dpi=300)
    plt.close()

    print(f"Actual vs predicted chart saved to: {ACTUAL_VS_PREDICTED_PATH}")


def create_error_distribution_chart(predictions_df: pd.DataFrame) -> None:
    """
    Create forecast error distribution chart.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(predictions_df["forecast_error"], bins=30)

    plt.title("Forecast Error Distribution")
    plt.xlabel("Forecast Error")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(ERROR_DISTRIBUTION_PATH, dpi=300)
    plt.close()

    print(f"Forecast error distribution chart saved to: {ERROR_DISTRIBUTION_PATH}")


def create_predicted_vs_actual_scatter(predictions_df: pd.DataFrame) -> None:
    """
    Create predicted vs actual scatter plot.
    """

    plt.figure(figsize=(8, 8))

    plt.scatter(
        predictions_df["units_sold"],
        predictions_df["predicted_units_sold"],
        alpha=0.5,
    )

    max_value = max(
        predictions_df["units_sold"].max(),
        predictions_df["predicted_units_sold"].max(),
    )

    plt.plot([0, max_value], [0, max_value], linestyle="--")

    plt.title("Predicted vs Actual Units Sold")
    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.tight_layout()

    plt.savefig(SCATTER_PLOT_PATH, dpi=300)
    plt.close()

    print(f"Predicted vs actual scatter plot saved to: {SCATTER_PLOT_PATH}")


def create_summary_statistics(predictions_df: pd.DataFrame) -> dict:
    """
    Create summary statistics for forecast output.
    """

    total_actual_demand = int(predictions_df["units_sold"].sum())
    total_predicted_demand = int(predictions_df["predicted_units_sold"].sum())
    average_error = round(float(predictions_df["forecast_error"].mean()), 4)
    absolute_error = abs(
        predictions_df["units_sold"] - predictions_df["predicted_units_sold"]
    )

    summary = {
        "total_prediction_rows": predictions_df.shape[0],
        "start_date": str(predictions_df["order_date"].min().date()),
        "end_date": str(predictions_df["order_date"].max().date()),
        "total_actual_demand": total_actual_demand,
        "total_predicted_demand": total_predicted_demand,
        "average_forecast_error": average_error,
        "average_absolute_error": round(float(absolute_error.mean()), 4),
        "max_absolute_error": int(absolute_error.max()),
    }

    return summary


def generate_html_report(
    metrics_df: pd.DataFrame,
    summary: dict,
    output_path: str = REPORT_PATH,
) -> None:
    """
    Generate HTML model evaluation report.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_model_row = metrics_df.sort_values("RMSE").iloc[0]
    best_model_name = best_model_row["model_name"]

    metrics_table = metrics_df.to_html(index=False)

    summary_table = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"],
    ).to_html(index=False)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Forecast Model Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f7f9fb;
                color: #222;
            }}
            h1 {{
                color: #1f4e79;
            }}
            h2 {{
                color: #333;
                margin-top: 30px;
            }}
            .summary-box {{
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #ddd;
                margin-bottom: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: #ffffff;
                margin-top: 15px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}
            th {{
                background-color: #1f4e79;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 10px;
                background-color: white;
            }}
        </style>
    </head>
    <body>
        <h1>Sales Demand Forecasting Project</h1>
        <h2>Model Evaluation Report</h2>

        <div class="summary-box">
            <p><strong>Generated At:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Best Model:</strong> {best_model_name}</p>
            <p><strong>Best RMSE:</strong> {best_model_row["RMSE"]}</p>
            <p><strong>Best MAE:</strong> {best_model_row["MAE"]}</p>
            <p><strong>Best MAPE:</strong> {best_model_row["MAPE"]}</p>
            <p><strong>Best R2:</strong> {best_model_row["R2"]}</p>
        </div>

        <h2>Model Metrics</h2>
        {metrics_table}

        <h2>Forecast Summary</h2>
        {summary_table}

        <h2>Model Performance Comparison</h2>
        <img src="model_metrics_comparison.png" alt="Model Metrics Comparison">

        <h2>Actual vs Predicted Demand</h2>
        <img src="actual_vs_predicted_demand.png" alt="Actual vs Predicted Demand">

        <h2>Forecast Error Distribution</h2>
        <img src="forecast_error_distribution.png" alt="Forecast Error Distribution">

        <h2>Predicted vs Actual Scatter Plot</h2>
        <img src="predicted_vs_actual_scatter.png" alt="Predicted vs Actual Scatter Plot">

        <h2>Interpretation</h2>
        <p>
            This report evaluates the forecasting model using standard regression metrics.
            Lower MAE, RMSE and MAPE values indicate better prediction accuracy.
            A higher R2 value indicates that the model explains more variation in sales demand.
        </p>

        <p>
            The actual vs predicted chart shows how closely the model follows real demand patterns.
            The error distribution shows whether prediction errors are centered around zero.
            The scatter plot shows whether predicted demand aligns with actual demand.
        </p>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Model evaluation HTML report saved to: {output_path}")


if __name__ == "__main__":
    print("Loading model outputs...")
    model_metrics_df, model_predictions_df = load_model_outputs()

    print("\nModel Metrics:")
    print(model_metrics_df)

    print("\nCreating evaluation charts...")
    create_metrics_comparison_chart(model_metrics_df)
    create_actual_vs_predicted_chart(model_predictions_df)
    create_error_distribution_chart(model_predictions_df)
    create_predicted_vs_actual_scatter(model_predictions_df)

    print("\nCreating forecast summary...")
    forecast_summary = create_summary_statistics(model_predictions_df)

    for key, value in forecast_summary.items():
        print(f"{key}: {value}")

    print("\nGenerating HTML model evaluation report...")
    generate_html_report(model_metrics_df, forecast_summary)

    print("\nModel evaluation completed successfully.")