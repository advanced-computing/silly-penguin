"""
load_to_bigquery.py

Pulls all EIA data sources and writes them to BigQuery.
Run locally with: python load_to_bigquery.py
Authenticate first: gcloud auth application-default login
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

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


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    """Fetch all data sources and load to BigQuery."""
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

    print("All done!")


if __name__ == "__main__":
    main()
