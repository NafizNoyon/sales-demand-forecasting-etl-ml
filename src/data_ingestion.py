import os
import numpy as np
import pandas as pd


RAW_DATA_PATH = "data/raw/sales_data.csv"


def create_demo_sales_data(output_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Create a realistic demo sales dataset for demand forecasting.

    The dataset includes daily sales records by store and product.
    Some missing values, duplicate rows, and outliers are intentionally added
    so that data quality validation can be demonstrated later.
    """

    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", end="2026-06-30", freq="D")

    stores = ["STORE_001", "STORE_002", "STORE_003"]
    products = [
        ("PROD_001", "Electronics", 650),
        ("PROD_002", "Grocery", 120),
        ("PROD_003", "Fashion", 350),
        ("PROD_004", "Home", 500),
        ("PROD_005", "Beauty", 220),
    ]

    regions = {
        "STORE_001": "Dhaka",
        "STORE_002": "Chattogram",
        "STORE_003": "Sylhet",
    }

    records = []

    for date in dates:
        day_of_week = date.dayofweek
        month = date.month

        weekend_effect = 1.25 if day_of_week in [4, 5] else 1.00
        seasonal_effect = 1.30 if month in [11, 12] else 1.00
        holiday = 1 if np.random.rand() < 0.06 else 0
        holiday_effect = 1.40 if holiday == 1 else 1.00

        for store in stores:
            for product_id, category, base_price in products:
                base_demand = np.random.randint(20, 80)

                demand = (
                    base_demand
                    * weekend_effect
                    * seasonal_effect
                    * holiday_effect
                    * np.random.normal(1.0, 0.15)
                )

                units_sold = max(1, int(demand))
                discount = round(np.random.choice([0.00, 0.05, 0.10, 0.15]), 2)
                marketing_spend = round(np.random.uniform(100, 1500), 2)
                unit_price = round(base_price * np.random.normal(1.0, 0.05), 2)
                revenue = round(units_sold * unit_price * (1 - discount), 2)

                records.append(
                    {
                        "order_date": date,
                        "store_id": store,
                        "product_id": product_id,
                        "category": category,
                        "region": regions[store],
                        "units_sold": units_sold,
                        "unit_price": unit_price,
                        "discount": discount,
                        "marketing_spend": marketing_spend,
                        "holiday": holiday,
                        "revenue": revenue,
                    }
                )

    df = pd.DataFrame(records)

    # Add missing values intentionally
    missing_indices = np.random.choice(df.index, size=40, replace=False)
    df.loc[missing_indices[:15], "units_sold"] = np.nan
    df.loc[missing_indices[15:30], "unit_price"] = np.nan
    df.loc[missing_indices[30:], "category"] = np.nan

    # Add duplicate rows intentionally
    duplicate_rows = df.sample(20, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Add outliers intentionally
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, "units_sold"] = df.loc[outlier_indices, "units_sold"] * 8

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def load_raw_sales_data(input_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw sales data from CSV file.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    df = pd.read_csv(input_path)
    return df


if __name__ == "__main__":
    print("Creating demo sales dataset...")
    created_df = create_demo_sales_data()

    print("Loading raw sales dataset...")
    loaded_df = load_raw_sales_data()

    print("\nDataset created successfully.")
    print(f"File path: {RAW_DATA_PATH}")
    print(f"Rows: {loaded_df.shape[0]}")
    print(f"Columns: {loaded_df.shape[1]}")

    print("\nColumn names:")
    print(list(loaded_df.columns))

    print("\nFirst 5 rows:")
    print(loaded_df.head())

    print("\nMissing values:")
    print(loaded_df.isnull().sum())

    print("\nDuplicate rows:")
    print(loaded_df.duplicated().sum())