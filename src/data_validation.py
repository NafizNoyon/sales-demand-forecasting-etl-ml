import os
import pandas as pd
from datetime import datetime


CLEANED_DATA_PATH = "data/processed/cleaned_sales_data.csv"
REPORT_PATH = "reports/data_quality_report.html"
SUMMARY_PATH = "reports/data_quality_summary.csv"


REQUIRED_COLUMNS = [
    "order_date",
    "store_id",
    "product_id",
    "category",
    "region",
    "units_sold",
    "unit_price",
    "discount",
    "marketing_spend",
    "holiday",
    "revenue",
]


def load_cleaned_data(input_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Load cleaned sales data from processed folder.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data file not found: {input_path}")

    df = pd.read_csv(input_path)
    return df


def add_validation_result(results: list, check_name: str, status: str, details: str) -> None:
    """
    Store validation check result.
    """

    results.append(
        {
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data quality validation checks on cleaned sales data.
    """

    results = []

    # Check 1: Required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if len(missing_columns) == 0:
        add_validation_result(
            results,
            "Required columns check",
            "PASS",
            "All required columns are present.",
        )
    else:
        add_validation_result(
            results,
            "Required columns check",
            "FAIL",
            f"Missing columns: {missing_columns}",
        )

    # Check 2: Missing values
    total_missing = int(df.isnull().sum().sum())

    if total_missing == 0:
        add_validation_result(
            results,
            "Missing values check",
            "PASS",
            "No missing values found in the dataset.",
        )
    else:
        add_validation_result(
            results,
            "Missing values check",
            "FAIL",
            f"Total missing values found: {total_missing}",
        )

    # Check 3: Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        add_validation_result(
            results,
            "Duplicate rows check",
            "PASS",
            "No duplicate rows found.",
        )
    else:
        add_validation_result(
            results,
            "Duplicate rows check",
            "FAIL",
            f"Duplicate rows found: {duplicate_count}",
        )

    # Check 4: Valid order_date
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    invalid_dates = int(df["order_date"].isnull().sum())

    if invalid_dates == 0:
        add_validation_result(
            results,
            "Order date validity check",
            "PASS",
            "All order_date values are valid dates.",
        )
    else:
        add_validation_result(
            results,
            "Order date validity check",
            "FAIL",
            f"Invalid order_date values found: {invalid_dates}",
        )

    # Check 5: units_sold should be non-negative
    invalid_units = int((df["units_sold"] < 0).sum())

    if invalid_units == 0:
        add_validation_result(
            results,
            "Units sold range check",
            "PASS",
            "All units_sold values are non-negative.",
        )
    else:
        add_validation_result(
            results,
            "Units sold range check",
            "FAIL",
            f"Negative units_sold values found: {invalid_units}",
        )

    # Check 6: unit_price should be positive
    invalid_price = int((df["unit_price"] <= 0).sum())

    if invalid_price == 0:
        add_validation_result(
            results,
            "Unit price range check",
            "PASS",
            "All unit_price values are positive.",
        )
    else:
        add_validation_result(
            results,
            "Unit price range check",
            "FAIL",
            f"Non-positive unit_price values found: {invalid_price}",
        )

    # Check 7: discount should be between 0 and 0.50
    invalid_discount = int(((df["discount"] < 0) | (df["discount"] > 0.50)).sum())

    if invalid_discount == 0:
        add_validation_result(
            results,
            "Discount range check",
            "PASS",
            "All discount values are between 0 and 0.50.",
        )
    else:
        add_validation_result(
            results,
            "Discount range check",
            "FAIL",
            f"Invalid discount values found: {invalid_discount}",
        )

    # Check 8: marketing_spend should be non-negative
    invalid_marketing = int((df["marketing_spend"] < 0).sum())

    if invalid_marketing == 0:
        add_validation_result(
            results,
            "Marketing spend range check",
            "PASS",
            "All marketing_spend values are non-negative.",
        )
    else:
        add_validation_result(
            results,
            "Marketing spend range check",
            "FAIL",
            f"Negative marketing_spend values found: {invalid_marketing}",
        )

    # Check 9: holiday should contain only 0 and 1
    invalid_holiday = int((~df["holiday"].isin([0, 1])).sum())

    if invalid_holiday == 0:
        add_validation_result(
            results,
            "Holiday binary check",
            "PASS",
            "Holiday column contains only 0 and 1.",
        )
    else:
        add_validation_result(
            results,
            "Holiday binary check",
            "FAIL",
            f"Invalid holiday values found: {invalid_holiday}",
        )

    # Check 10: revenue should be non-negative
    invalid_revenue = int((df["revenue"] < 0).sum())

    if invalid_revenue == 0:
        add_validation_result(
            results,
            "Revenue range check",
            "PASS",
            "All revenue values are non-negative.",
        )
    else:
        add_validation_result(
            results,
            "Revenue range check",
            "FAIL",
            f"Negative revenue values found: {invalid_revenue}",
        )

    result_df = pd.DataFrame(results)
    return result_df


def create_data_profile(df: pd.DataFrame) -> dict:
    """
    Create basic data profile summary for reporting.
    """

    profile = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "total_missing_values": int(df.isnull().sum().sum()),
        "total_duplicate_rows": int(df.duplicated().sum()),
        "start_date": str(pd.to_datetime(df["order_date"]).min().date()),
        "end_date": str(pd.to_datetime(df["order_date"]).max().date()),
        "unique_stores": df["store_id"].nunique(),
        "unique_products": df["product_id"].nunique(),
        "unique_categories": df["category"].nunique(),
        "total_units_sold": int(df["units_sold"].sum()),
        "total_revenue": round(float(df["revenue"].sum()), 2),
    }

    return profile


def generate_html_report(
    validation_results: pd.DataFrame,
    profile: dict,
    output_path: str = REPORT_PATH,
) -> None:
    """
    Generate an HTML data quality report.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pass_count = int((validation_results["status"] == "PASS").sum())
    fail_count = int((validation_results["status"] == "FAIL").sum())

    overall_status = "PASS" if fail_count == 0 else "FAIL"

    validation_table = validation_results.to_html(index=False, escape=False)

    profile_table = pd.DataFrame(
        list(profile.items()),
        columns=["Metric", "Value"],
    ).to_html(index=False)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sales Data Quality Report</title>
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
            .pass {{
                color: green;
                font-weight: bold;
            }}
            .fail {{
                color: red;
                font-weight: bold;
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
        </style>
    </head>
    <body>
        <h1>Sales Demand Forecasting Project</h1>
        <h2>Data Quality Validation Report</h2>

        <div class="summary-box">
            <p><strong>Generated At:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Overall Status:</strong> <span class="{overall_status.lower()}">{overall_status}</span></p>
            <p><strong>Passed Checks:</strong> {pass_count}</p>
            <p><strong>Failed Checks:</strong> {fail_count}</p>
        </div>

        <h2>Dataset Profile</h2>
        {profile_table}

        <h2>Validation Results</h2>
        {validation_table}

        <h2>Interpretation</h2>
        <p>
            This report validates the cleaned sales dataset before using it for demand forecasting.
            The checks confirm whether the data is complete, consistent, duplicate-free and suitable
            for machine learning model training.
        </p>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"HTML data quality report saved to: {output_path}")


def save_validation_summary(
    validation_results: pd.DataFrame,
    output_path: str = SUMMARY_PATH,
) -> None:
    """
    Save validation summary as CSV.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    validation_results.to_csv(output_path, index=False)
    print(f"Validation summary saved to: {output_path}")


if __name__ == "__main__":
    print("Loading cleaned sales data...")
    sales_df = load_cleaned_data()

    print("Running data quality validation checks...")
    validation_df = validate_sales_data(sales_df)

    print("\nValidation Results:")
    print(validation_df)

    profile_summary = create_data_profile(sales_df)

    print("\nDataset Profile:")
    for key, value in profile_summary.items():
        print(f"{key}: {value}")

    save_validation_summary(validation_df)
    generate_html_report(validation_df, profile_summary)

    failed_checks = validation_df[validation_df["status"] == "FAIL"]

    if failed_checks.empty:
        print("\nData quality validation completed successfully.")
        print("All checks passed.")
    else:
        print("\nData quality validation completed with failed checks.")
        print(failed_checks)