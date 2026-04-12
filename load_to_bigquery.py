"""load_to_bigquery.py

Runs the full ETL: fetch all raw data sources, write to BigQuery,
then build pre-aggregated summary tables for fast dashboard loading.

Sources:
  EIA (demand, interchange, fuel, NG prices) — requires EIA_API_KEY
  Open-Meteo (weather)                       — no key
  gridstatus (ISO LMP: PJM/CAISO/ERCOT)      — no key (CAISO/ERCOT);
                                                PJM_API_KEY optional
  NREL (solar resource)                      — requires NREL_API_KEY
  LBNL (interconnection queue)               — local Excel file

Run locally with: python load_to_bigquery.py
Authenticate GCP first: gcloud auth application-default login
"""

from __future__ import annotations

import os
import traceback

from dotenv import load_dotenv
from google.cloud import bigquery

from data_fetching import (
    fetch_demand_data,
    fetch_fuel_type_data,
    fetch_interchange_data,
    fetch_iso_lmp,
    fetch_natural_gas_prices,
    fetch_nrel_resources,
    fetch_weather_data,
    load_lbnl_queue,
)

# --------------------------------------------------
# Config
# --------------------------------------------------
load_dotenv()
EIA_API_KEY = os.getenv("EIA_API_KEY")
NREL_API_KEY = os.getenv("NREL_API_KEY")

GCP_PROJECT = "sipa-adv-c-silly-penguin"
BQ_DATASET = "eia_data"


# --------------------------------------------------
# Generic writer
# --------------------------------------------------
def write_to_bq(df, table_name: str) -> bool:
    """Write a DataFrame to BigQuery, replacing existing data."""
    if df is None or df.empty:
        print(f"  Skipped {table_name}: dataframe is empty")
        return False

    destination = f"{BQ_DATASET}.{table_name}"
    print(f"  Writing {len(df)} rows to {GCP_PROJECT}.{destination}")
    print(f"  Columns: {df.columns.tolist()}")

    try:
        df.to_gbq(
            destination_table=destination,
            project_id=GCP_PROJECT,
            if_exists="replace",
        )
        print(f"  ✓ Wrote {len(df)} rows to {GCP_PROJECT}.{destination}")
        return True
    except Exception as e:
        print(f"  ❌ Failed writing to {destination}: {e}")
        return False


# --------------------------------------------------
# Aggregation SQL
# --------------------------------------------------
def run_aggregation_sql(client: bigquery.Client, tables_present: set) -> None:
    """Create pre-aggregated tables. Each aggregation is independent:
    if its source table is missing, it's skipped.
    """
    dataset = f"{GCP_PROJECT}.{BQ_DATASET}"

    if "hourly_demand" in tables_present:
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

    if "hourly_interchange" in tables_present:
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

    if "hourly_fuel_type" in tables_present:
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

    if "hourly_demand" in tables_present:
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
                    SELECT
                        period,
                        respondent,
                        `type-name` AS type_name,
                        value AS avg_value
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

    if "iso_hourly_lmp" in tables_present:
        print("  Creating iso_lmp_daily_stats...")
        client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.iso_lmp_daily_stats` AS
            SELECT
                iso,
                location,
                location_type,
                DATE(time) AS date,
                AVG(lmp) AS avg_lmp,
                MAX(lmp) AS peak_lmp,
                MIN(lmp) AS min_lmp,
                STDDEV(lmp) AS lmp_volatility,
                AVG(IF(EXTRACT(HOUR FROM time) BETWEEN 14 AND 19, lmp, NULL)) AS peak_hours_avg_lmp,
                AVG(congestion) AS avg_congestion,
                COUNT(*) AS n_hours
            FROM `{dataset}.iso_hourly_lmp`
            GROUP BY iso, location, location_type, date
        """).result()

    if "interconnection_queue" in tables_present:
        print("  Creating queue_ba_summary...")
        client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.queue_ba_summary` AS
            SELECT
                ba,
                COUNT(*) AS total_projects,
                COUNTIF(q_status = 'active') AS active_projects,
                COUNTIF(q_status = 'operational') AS operational_projects,
                COUNTIF(q_status = 'withdrawn') AS withdrawn_projects,
                SUM(IF(q_status = 'active', mw1, 0)) AS active_capacity_mw,
                SUM(IF(q_status = 'operational', mw1, 0)) AS operational_capacity_mw,
                AVG(IF(q_status = 'active', mw1, NULL)) AS avg_project_size_mw,
                SAFE_DIVIDE(
                    COUNTIF(q_status = 'operational'),
                    COUNTIF(q_status IN ('operational', 'withdrawn'))
                ) AS completion_rate
            FROM `{dataset}.interconnection_queue`
            WHERE ba IS NOT NULL
            GROUP BY ba
        """).result()

        print("  Creating queue_type_summary...")
        client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.queue_type_summary` AS
            SELECT
                ba,
                type_clean AS resource_type,
                COUNT(*) AS n_projects,
                SUM(IF(q_status = 'active', mw1, 0)) AS active_mw,
                SUM(IF(q_status = 'operational', mw1, 0)) AS operational_mw
            FROM `{dataset}.interconnection_queue`
            WHERE ba IS NOT NULL AND type_clean IS NOT NULL
            GROUP BY ba, resource_type
        """).result()

    print("  ✓ Aggregation step complete")


# --------------------------------------------------
# Main
# --------------------------------------------------
def _try_fetch_and_write(label: str, fetch_fn, table_name: str, results: dict) -> None:
    """Run one fetch+write step, isolating its failures."""
    print(f"\n--- {label} ---")
    try:
        df = fetch_fn()
        print(f"  Rows fetched: {len(df) if df is not None else 0}")
        success = write_to_bq(df, table_name)
        results[table_name] = success
    except Exception as e:
        print(f"  ❌ {label} step failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        results[table_name] = False


def main() -> None:
    """Run full ETL with isolated per-source failure handling."""
    if not EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY not found in environment.")

    results: dict[str, bool] = {}

    _try_fetch_and_write(
        "1/8 EIA demand",
        lambda: fetch_demand_data(EIA_API_KEY),
        "hourly_demand",
        results,
    )
    _try_fetch_and_write(
        "2/8 EIA interchange",
        lambda: fetch_interchange_data(EIA_API_KEY),
        "hourly_interchange",
        results,
    )
    _try_fetch_and_write(
        "3/8 EIA fuel type",
        lambda: fetch_fuel_type_data(EIA_API_KEY),
        "hourly_fuel_type",
        results,
    )
    _try_fetch_and_write(
        "4/8 EIA NG price",
        lambda: fetch_natural_gas_prices(EIA_API_KEY),
        "daily_ng_price",
        results,
    )
    _try_fetch_and_write(
        "5/8 Open-Meteo weather",
        fetch_weather_data,
        "daily_weather",
        results,
    )
    _try_fetch_and_write(
        "6/8 gridstatus ISO LMP",
        fetch_iso_lmp,
        "iso_hourly_lmp",
        results,
    )
    _try_fetch_and_write(
        "7/8 NREL solar resource",
        lambda: fetch_nrel_resources(NREL_API_KEY or ""),
        "nrel_resource_locations",
        results,
    )
    _try_fetch_and_write(
        "8/8 LBNL interconnection queue",
        load_lbnl_queue,
        "interconnection_queue",
        results,
    )

    print("\n" + "=" * 50)
    print("Raw write summary:")
    for table, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {table}")
    print("=" * 50)

    tables_present = {k for k, v in results.items() if v}
    if not tables_present:
        raise RuntimeError(
            "No raw tables were successfully written; nothing to aggregate."
        )

    print("\n--- Running aggregations ---")
    client = bigquery.Client(project=GCP_PROJECT)
    run_aggregation_sql(client, tables_present)

    print("\nAll done!")


if __name__ == "__main__":
    main()
