"""
load_to_bigquery.py

Pulls all EIA data sources, writes raw tables to BigQuery,
then creates pre-aggregated daily tables for fast dashboard loading.

Run locally with: python load_to_bigquery.py
Authenticate first: gcloud auth application-default login
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.cloud import bigquery

from data_fetching import (
    fetch_demand_data,
    fetch_fuel_type_data,
    fetch_interchange_data,
    fetch_natural_gas_prices,
    fetch_weather_data,
)

# --------------------------------------------------
# Config
# --------------------------------------------------
load_dotenv()
EIA_API_KEY = os.getenv("EIA_API_KEY")

GCP_PROJECT = "sipa-adv-c-silly-penguin"
BQ_DATASET = "eia_data"


# --------------------------------------------------
# Write to BigQuery
# --------------------------------------------------
def write_to_bq(df, table_name: str) -> None:
    """Write a DataFrame to BigQuery, replacing existing data."""
    destination = f"{BQ_DATASET}.{table_name}"
    df.to_gbq(
        destination_table=destination,
        project_id=GCP_PROJECT,
        if_exists="replace",
    )
    print(f"  Wrote {len(df)} rows to {GCP_PROJECT}.{destination}")


def run_aggregation_sql(client: bigquery.Client) -> None:
    """Create pre-aggregated daily tables from raw hourly data."""
    dataset = f"{GCP_PROJECT}.{BQ_DATASET}"

    # Daily demand summary: one row per (date, respondent)
    print("  Creating daily_demand_summary...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{dataset}.daily_demand_summary` AS
        SELECT
            DATE(period) AS date,
            respondent,
            `type-name` AS type_name,
            AVG(value) AS avg_value,
            MAX(value) AS max_value,
            MIN(value) AS min_value,
            COUNT(*) AS n_hours
        FROM `{dataset}.hourly_demand`
        GROUP BY date, respondent, type_name
    """).result()

    # Daily interchange summary: one row per (date, fromba)
    print("  Creating daily_interchange_summary...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{dataset}.daily_interchange_summary` AS
        SELECT
            DATE(period) AS date,
            fromba,
            SUM(value) AS total_interchange,
            AVG(value) AS avg_interchange,
            MAX(value) AS max_interchange,
            MIN(value) AS min_interchange
        FROM `{dataset}.hourly_interchange`
        GROUP BY date, fromba
    """).result()

    # Daily fuel mix summary: one row per (date, respondent, fueltype)
    print("  Creating daily_fuel_summary...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{dataset}.daily_fuel_summary` AS
        SELECT
            DATE(period) AS date,
            respondent,
            fueltype,
            AVG(value) AS avg_generation,
            SUM(value) AS total_generation
        FROM `{dataset}.hourly_fuel_type`
        GROUP BY date, respondent, fueltype
    """).result()

    # BA MAPE ranking: one row per respondent
    print("  Creating ba_mape_ranking...")
    client.query(f"""
        CREATE OR REPLACE TABLE `{dataset}.ba_mape_ranking` AS
        WITH pivoted AS (
            SELECT
                period,
                respondent,
                MAX(IF(type_name = 'Demand', avg_value, NULL)) AS demand,
                MAX(IF(type_name = 'Day-ahead demand forecast', avg_value, NULL)) AS forecast
            FROM (
                SELECT period, respondent, `type-name` AS type_name, value AS avg_value
                FROM `{dataset}.hourly_demand`
            )
            GROUP BY period, respondent
        )
        SELECT
            respondent AS ba,
            AVG(ABS(demand - forecast) / NULLIF(demand, 0) * 100) AS mape,
            AVG(ABS(demand - forecast)) AS mae,
            AVG(demand) AS avg_demand,
            COUNT(*) AS n_hours
        FROM pivoted
        WHERE demand IS NOT NULL AND forecast IS NOT NULL
        GROUP BY respondent
        ORDER BY mape
    """).result()

    print("  Aggregation complete!")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    """Fetch all data sources, load to BigQuery, then aggregate."""
    if not EIA_API_KEY:
        msg = "EIA_API_KEY not found in environment. Set it in .env file."
        raise RuntimeError(msg)

    # 1) Demand & Forecast
    print("1/5  Fetching demand data...")
    demand_df = fetch_demand_data(EIA_API_KEY)
    if not demand_df.empty:
        write_to_bq(demand_df, "hourly_demand")

    # 2) Interchange
    print("2/5  Fetching interchange data...")
    interchange_df = fetch_interchange_data(EIA_API_KEY)
    if not interchange_df.empty:
        write_to_bq(interchange_df, "hourly_interchange")

    # 3) Generation by Fuel Type
    print("3/5  Fetching fuel type data...")
    fuel_df = fetch_fuel_type_data(EIA_API_KEY)
    if not fuel_df.empty:
        write_to_bq(fuel_df, "hourly_fuel_type")

    # 4) Natural Gas Prices
    print("4/5  Fetching natural gas prices...")
    ng_price_df = fetch_natural_gas_prices(EIA_API_KEY)
    if not ng_price_df.empty:
        write_to_bq(ng_price_df, "daily_ng_price")

    # 5) Weather
    print("5/5  Fetching weather data...")
    weather_df = fetch_weather_data()
    if not weather_df.empty:
        write_to_bq(weather_df, "daily_weather")

    # 6) Pre-aggregate
    print("6/6  Building aggregated tables...")
    client = bigquery.Client(project=GCP_PROJECT)
    run_aggregation_sql(client)

    print("All done!")


if __name__ == "__main__":
    main()
