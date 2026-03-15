"""
load_to_bigquery.py
Pulls EIA hourly demand data and writes it to BigQuery.
Run locally with: python load_to_bigquery.py
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# Config
# --------------------------------------------------
load_dotenv()
EIA_API_KEY = os.getenv("EIA_API_KEY")

# Replace with YOUR GCP project ID and dataset
GCP_PROJECT = "sipa-adv-c-silly-penguin"  # e.g. "grid-monitor-123456"
BQ_DATASET = "eia_data"  # dataset name in BigQuery
BQ_TABLE = f"{BQ_DATASET}.hourly_demand"  # dataset.table

HTTP_OK = 200


# --------------------------------------------------
# 1) Pull data from EIA API
# --------------------------------------------------
def fetch_eia_data():
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": ["D", "DF"],
        "start": "2025-01-01T00",
        "end": "2026-02-01T00",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != HTTP_OK:
        raise RuntimeError(f"EIA API returned status {response.status_code}")

    data = response.json()["response"]["data"]
    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df


# --------------------------------------------------
# 2) Write to BigQuery (creates table if not exists)
# --------------------------------------------------
def load_to_bigquery(df):
    """
    Uses pandas-gbq to write the DataFrame to BigQuery.

    if_exists options:
      - "fail"    : raise error if table exists
      - "replace" : drop + recreate table each run
      - "append"  : add rows to existing table

    For a regularly-updated source, "replace" is simplest
    (full refresh each run). Use "append" if you want to
    accumulate historical data and handle deduplication yourself.
    """
    df.to_gbq(
        destination_table=BQ_TABLE,
        project_id=GCP_PROJECT,
        if_exists="replace",  # <-- technique: full replace each run
    )
    print(f"✅ Wrote {len(df)} rows to {GCP_PROJECT}.{BQ_TABLE}")


# --------------------------------------------------
# 3) Main
# --------------------------------------------------
if __name__ == "__main__":
    print("Fetching EIA data...")
    df = fetch_eia_data()
    print(f"Fetched {len(df)} rows.")

    print("Loading to BigQuery...")
    load_to_bigquery(df)
