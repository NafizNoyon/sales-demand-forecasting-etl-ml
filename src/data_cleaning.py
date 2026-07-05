import os
import pandas as pd


RAW_DATA_PATH = "data/raw/sales_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_sales_data.csv"


def load_raw_data(input_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw sales data from CSV file.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    df = pd.read_csv(input_path)
    return df


def fill_missing_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing category values using product_id based category mapping.
    """

    df = df.copy()

    product_category_map = (
        df.dropna(subset=["category"])
        .groupby("product_id")["category"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    df["category"] = df["category"].fillna(df["product_id"].map(product_category_map))
    df["category"] = df["category"].fillna("Unknown")

    return df


def fill_missing_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values using grouped median values.
    """

    df = df.copy()

    df["units_sold"] = df["units_sold"].fillna(
        df.groupby(["product_id", "store_id"])["units_sold"].transform("median")
    )

    df["unit_price"] = df["unit_price"].fillna(
        df.groupby("product_id")["unit_price"].transform("median")
    )

    df["units_sold"] = df["units_sold"].fillna(df["units_sold"].median())
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

    return df


def cap_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cap outliers using the IQR method.
    Values below lower limit are replaced with lower limit.
    Values above upper limit are replaced with upper limit.
    """

    df = df.copy()

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)

    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw sales data for demand forecasting.
    """

    df = df.copy()

    print("Starting data cleaning process...")
    print(f"Initial rows: {df.shape[0]}")
    print(f"Initial columns: {df.shape[1]}")

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {duplicate_count}")

    # Convert order_date to datetime
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Remove rows where order_date is invalid
    invalid_date_count = df["order_date"].isna().sum()
    df = df.dropna(subset=["order_date"])
    print(f"Invalid date rows removed: {invalid_date_count}")

    # Fill missing category
    df = fill_missing_category(df)

    # Fill missing numeric values
    df = fill_missing_numeric_values(df)

    # Handle outliers
    df = cap_outliers_iqr(df, "units_sold")
    df = cap_outliers_iqr(df, "unit_price")
    df = cap_outliers_iqr(df, "marketing_spend")

    # Convert units_sold to integer
    df["units_sold"] = df["units_sold"].round().astype(int)

    # Recalculate revenue after cleaning
    df["revenue"] = (
        df["units_sold"] * df["unit_price"] * (1 - df["discount"])
    ).round(2)

    # Sort data
    df = df.sort_values(by=["order_date", "store_id", "product_id"]).reset_index(drop=True)

    print("\nData cleaning completed.")
    print(f"Final rows: {df.shape[0]}")
    print(f"Final columns: {df.shape[1]}")

    return df


def save_clean_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> None:
    """
    Save cleaned sales data to processed folder.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")


if __name__ == "__main__":
    raw_df = load_raw_data()

    print("\nBefore cleaning missing values:")
    print(raw_df.isnull().sum())

    print("\nBefore cleaning duplicate rows:")
    print(raw_df.duplicated().sum())

    cleaned_df = clean_sales_data(raw_df)

    print("\nAfter cleaning missing values:")
    print(cleaned_df.isnull().sum())

    print("\nAfter cleaning duplicate rows:")
    print(cleaned_df.duplicated().sum())

    print("\nCleaned data sample:")
    print(cleaned_df.head())

    save_clean_data(cleaned_df)