import os
import pandas as pd
import plotly.express as px
import streamlit as st

PREDICTIONS_PATH = "data/forecast_output/model_predictions.csv"
METRICS_PATH = "reports/model_metrics.csv"
DATA_QUALITY_PATH = "reports/data_quality_summary.csv"

st.set_page_config(
    page_title="Sales Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)

@st.cache_data
def load_data_quality():
    if os.path.exists(DATA_QUALITY_PATH):
        return pd.read_csv(DATA_QUALITY_PATH)
    return pd.DataFrame()

st.title("Sales Demand Forecasting Dashboard")
st.write(
    "This dashboard shows sales demand forecasting results, model performance, "
    "forecast errors, and data quality validation summary."
)

if not os.path.exists(PREDICTIONS_PATH):
    st.error(f"Prediction file not found: {PREDICTIONS_PATH}")
    st.stop()

if not os.path.exists(METRICS_PATH):
    st.error(f"Model metrics file not found: {METRICS_PATH}")
    st.stop()

predictions_df = load_predictions()
metrics_df = load_metrics()
quality_df = load_data_quality()

st.sidebar.header("Filters")

store_list = sorted(predictions_df["store_id"].unique())
product_list = sorted(predictions_df["product_id"].unique())
category_list = sorted(predictions_df["category"].unique())
region_list = sorted(predictions_df["region"].unique())

selected_stores = st.sidebar.multiselect("Select Store", store_list, default=store_list)
selected_products = st.sidebar.multiselect("Select Product", product_list, default=product_list)
selected_categories = st.sidebar.multiselect("Select Category", category_list, default=category_list)
selected_regions = st.sidebar.multiselect("Select Region", region_list, default=region_list)

filtered_df = predictions_df[
    predictions_df["store_id"].isin(selected_stores)
    & predictions_df["product_id"].isin(selected_products)
    & predictions_df["category"].isin(selected_categories)
    & predictions_df["region"].isin(selected_regions)
].copy()

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

best_model = metrics_df.sort_values("RMSE").iloc[0]

total_actual = int(filtered_df["units_sold"].sum())
total_predicted = int(filtered_df["predicted_units_sold"].sum())
average_absolute_error = abs(
    filtered_df["units_sold"] - filtered_df["predicted_units_sold"]
).mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Actual Demand", f"{total_actual:,}")
col2.metric("Total Predicted Demand", f"{total_predicted:,}")
col3.metric("Best Model", best_model["model_name"])
col4.metric("Avg Absolute Error", round(average_absolute_error, 3))

st.divider()

st.subheader("Model Performance Metrics")
st.dataframe(metrics_df, use_container_width=True)

metrics_long = metrics_df.melt(
    id_vars="model_name",
    value_vars=["MAE", "RMSE", "MAPE", "R2"],
    var_name="Metric",
    value_name="Value"
)

fig_metrics = px.bar(
    metrics_long,
    x="model_name",
    y="Value",
    color="Metric",
    barmode="group",
    title="Model Metrics Comparison"
)
st.plotly_chart(fig_metrics, use_container_width=True)

st.divider()

st.subheader("Actual vs Predicted Demand")

daily_df = (
    filtered_df.groupby("order_date")[["units_sold", "predicted_units_sold"]]
    .sum()
    .reset_index()
)

daily_long = daily_df.melt(
    id_vars="order_date",
    value_vars=["units_sold", "predicted_units_sold"],
    var_name="Demand Type",
    value_name="Units Sold"
)

fig_daily = px.line(
    daily_long,
    x="order_date",
    y="Units Sold",
    color="Demand Type",
    markers=True,
    title="Daily Actual vs Predicted Demand"
)
st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

col5, col6 = st.columns(2)

with col5:
    st.subheader("Product-Level Demand")
    product_df = (
        filtered_df.groupby("product_id")[["units_sold", "predicted_units_sold"]]
        .sum()
        .reset_index()
    )
    product_long = product_df.melt(
        id_vars="product_id",
        value_vars=["units_sold", "predicted_units_sold"],
        var_name="Demand Type",
        value_name="Units Sold"
    )
    fig_product = px.bar(
        product_long,
        x="product_id",
        y="Units Sold",
        color="Demand Type",
        barmode="group",
        title="Actual vs Predicted by Product"
    )
    st.plotly_chart(fig_product, use_container_width=True)

with col6:
    st.subheader("Store-Level Demand")
    store_df = (
        filtered_df.groupby("store_id")[["units_sold", "predicted_units_sold"]]
        .sum()
        .reset_index()
    )
    store_long = store_df.melt(
        id_vars="store_id",
        value_vars=["units_sold", "predicted_units_sold"],
        var_name="Demand Type",
        value_name="Units Sold"
    )
    fig_store = px.bar(
        store_long,
        x="store_id",
        y="Units Sold",
        color="Demand Type",
        barmode="group",
        title="Actual vs Predicted by Store"
    )
    st.plotly_chart(fig_store, use_container_width=True)

st.divider()

st.subheader("Forecast Error Analysis")

col7, col8 = st.columns(2)

with col7:
    fig_error = px.histogram(
        filtered_df,
        x="forecast_error",
        nbins=30,
        title="Forecast Error Distribution"
    )
    st.plotly_chart(fig_error, use_container_width=True)

with col8:
    fig_scatter = px.scatter(
        filtered_df,
        x="units_sold",
        y="predicted_units_sold",
        color="category",
        title="Predicted vs Actual Units Sold"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

st.subheader("Data Quality Validation Summary")

if quality_df.empty:
    st.warning("Data quality summary file not found.")
else:
    st.dataframe(quality_df, use_container_width=True)

st.divider()

st.subheader("Forecast Output Table")

display_columns = [
    "order_date",
    "store_id",
    "product_id",
    "category",
    "region",
    "units_sold",
    "predicted_units_sold",
    "forecast_error"
]

st.dataframe(
    filtered_df[display_columns].sort_values("order_date", ascending=False),
    use_container_width=True
)
