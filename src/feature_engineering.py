import os
import pandas as pd


CLEANED_DATA_PATH = "data/processed/cleaned_sales_data.csv"
FEATURE_DATA_PATH = "data/processed/feature_sales_data.csv"


def load_cleaned_data(input_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Load cleaned sales data from processed folder.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data file not found: {input_path}")

    df = pd.read_csv(input_path)
    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date-based features from order_date.
    """

    df = df.copy()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["order_date"].dt.quarter

    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)
    df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for units_sold and revenue.
    Lag features help the model learn from previous demand patterns.
    """

    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "order_date"])

    group_cols = ["store_id", "product_id"]

    lag_days = [1, 7, 14, 28]

    for lag in lag_days:
        df[f"units_sold_lag_{lag}"] = (
            df.groupby(group_cols)["units_sold"].shift(lag)
        )

        df[f"revenue_lag_{lag}"] = (
            df.groupby(group_cols)["revenue"].shift(lag)
        )

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling average features.
    Rolling features help capture recent sales trends.
    """

    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "order_date"])

    group_cols = ["store_id", "product_id"]
    rolling_windows = [7, 14, 28]

    for window in rolling_windows:
        df[f"units_sold_rolling_mean_{window}"] = (
            df.groupby(group_cols)["units_sold"]
            .transform(lambda x: x.shift(1).rolling(window=window).mean())
        )

        df[f"units_sold_rolling_std_{window}"] = (
            df.groupby(group_cols)["units_sold"]
            .transform(lambda x: x.shift(1).rolling(window=window).std())
        )

        df[f"revenue_rolling_mean_{window}"] = (
            df.groupby(group_cols)["revenue"]
            .transform(lambda x: x.shift(1).rolling(window=window).mean())
        )

    return df


def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-related features for forecasting.
    """

    df = df.copy()

    df["price_after_discount"] = df["unit_price"] * (1 - df["discount"])

    df["revenue_per_unit"] = df["revenue"] / df["units_sold"]
    df["revenue_per_unit"] = df["revenue_per_unit"].replace([float("inf"), -float("inf")], 0)

    df["marketing_spend_per_unit"] = df["marketing_spend"] / df["units_sold"]
    df["marketing_spend_per_unit"] = df["marketing_spend_per_unit"].replace(
        [float("inf"), -float("inf")],
        0,
    )

    return df


def create_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature dataset for sales demand forecasting.
    """

    print("Starting feature engineering process...")
    print(f"Input rows: {df.shape[0]}")
    print(f"Input columns: {df.shape[1]}")

    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_business_features(df)

    # Drop rows with missing values created by lag and rolling features
    before_drop = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    after_drop = df.shape[0]

    print(f"Rows removed due to lag or rolling missing values: {before_drop - after_drop}")

    # Sort final dataset
    df = df.sort_values(by=["order_date", "store_id", "product_id"]).reset_index(drop=True)

    print("\nFeature engineering completed.")
    print(f"Final rows: {df.shape[0]}")
    print(f"Final columns: {df.shape[1]}")

    return df


def save_feature_data(df: pd.DataFrame, output_path: str = FEATURE_DATA_PATH) -> None:
    """
    Save feature engineered dataset.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeature data saved to: {output_path}")


if __name__ == "__main__":
    print("Loading cleaned sales data...")
    sales_df = load_cleaned_data()

    feature_df = create_forecasting_features(sales_df)

    print("\nFeature dataset sample:")
    print(feature_df.head())

    print("\nFeature dataset columns:")
    print(list(feature_df.columns))

    save_feature_data(feature_df)