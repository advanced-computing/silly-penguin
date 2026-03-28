# Data Loading Strategy

## Overview

Our app uses a **two-stage architecture**: a batch ETL pipeline writes data to Google BigQuery, and the Streamlit dashboard reads from BigQuery at startup, caching everything in memory for fast page navigation.

## Stage 1: ETL Pipeline (`load_to_bigquery.py`)

We run `load_to_bigquery.py` manually to pull data from external APIs and write it to BigQuery. This script:

1. Fetches hourly data from **3 EIA API endpoints** (demand/forecast, interchange, generation by fuel type), plus daily natural gas prices and weather data
2. Writes raw data to **5 BigQuery tables** (`hourly_demand`, `hourly_interchange`, `hourly_fuel_type`, `daily_ng_price`, `daily_weather`)
3. Runs SQL aggregation queries to create **4 pre-computed summary tables** (`daily_demand_summary`, `daily_interchange_summary`, `daily_fuel_summary`, `ba_mape_ranking`)

We chose `if_exists="replace"` (full refresh) over `"append"` because our time window is a rolling 3-month period. A full refresh is simpler and avoids deduplication logic.

## Stage 2: Streamlit App (`app.py`)

The app loads **all data once at startup** using `st.cache_resource`, then every page reads from the in-memory cache with zero network latency.

```python
@st.cache_resource(show_spinner="Loading data from BigQuery (one-time)...")
def _load_all_data():
    # Runs 6 BigQuery queries, returns a dict of DataFrames
    ...

_data = _load_all_data()
```

### Why `st.cache_resource` instead of `st.cache_data`?

We tested both. `st.cache_data` serializes and deserializes DataFrames on every rerun, adding 1–2 seconds per page. `st.cache_resource` keeps the Python objects in memory directly, so subsequent page loads are < 0.5 seconds.

### Why load everything upfront instead of per-page?

BigQuery has high per-query latency (~2–3 seconds) due to network round-trips and job scheduling overhead. We tested several approaches:

| Approach | Cold start | Page switch | Verdict |
|----------|-----------|-------------|---------|
| Load all data on every page | ~7s per page | ~7s | Too slow |
| Lazy load per page with `cache_data` | ~5s first visit | <2s cached | First visit too slow |
| Per-BA filtered queries | ~3–5s per page | ~3–5s | Still exceeds 2s target |
| Pre-aggregated daily tables | ~6s (4 queries) | <1s cached | Query overhead still high |
| **Upfront load with `cache_resource`** | **~10s once** | **< 0.5s** | **Best UX after first load** |

The upfront approach pays a one-time cost when the app starts, but delivers instant page navigation afterward. Since users typically browse multiple pages per session, this gives the best overall experience.

### Why BigQuery instead of direct API calls?

The EIA API is rate-limited and slow for large queries (10 BAs × 3 months of hourly data requires many paginated requests). BigQuery serves as a fast intermediate store: the ETL pipeline handles the slow API pagination once, and the app reads pre-loaded data quickly.

## Data Flow Diagram

```
EIA API ──→ load_to_bigquery.py ──→ BigQuery (raw + aggregated tables)
                                          │
Open-Meteo ──→ load_to_bigquery.py ───────┘
                                          │
                                          ▼
                                    Streamlit app
                                  (st.cache_resource)
                                          │
                                          ▼
                                   In-memory DataFrames
                                   (all pages read from here)
```
