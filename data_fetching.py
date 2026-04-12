"""Data fetching functions for all EIA and weather data sources.

Key optimization: EIA API accepts multiple facet values in one request
(e.g. facets[respondent][]=CISO&facets[respondent][]=ERCO&...) so we
batch all BAs into a single paginated call instead of looping one-by-one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

import time

HTTP_OK = 200

# ---------------------------------------------------------------------------
# Major Balancing Authorities we track
# ---------------------------------------------------------------------------
MAJOR_BA = [
    "CISO",  # California ISO
    "ERCO",  # ERCOT (Texas)
    "PJM",  # PJM Interconnection
    "MISO",  # Midcontinent ISO
    "NYIS",  # New York ISO
    "ISNE",  # ISO New England
    "SWPP",  # Southwest Power Pool
    "SOCO",  # Southern Company
    "TVA",  # Tennessee Valley Authority
    "BPAT",  # Bonneville Power Administration
]

# Representative city coordinates for weather data per BA
BA_WEATHER_COORDS: dict[str, tuple[float, float]] = {
    "CISO": (36.78, -119.42),
    "ERCO": (31.97, -99.90),
    "PJM": (39.95, -75.16),
    "MISO": (41.88, -87.63),
    "NYIS": (40.71, -74.01),
    "ISNE": (42.36, -71.06),
    "SWPP": (35.47, -97.52),
    "SOCO": (33.75, -84.39),
    "TVA": (36.16, -86.78),
    "BPAT": (45.52, -122.68),
}

# EIA fuel type codes
FUEL_TYPES = ["NG", "SUN", "WND", "NUC", "COL", "WAT", "OIL", "OTH"]

# Time range: always "3 months ago → today", computed dynamically
_today = datetime.now()
_three_months_ago = _today - timedelta(days=90)

DEFAULT_START = _three_months_ago.strftime("%Y-%m-%dT00")
DEFAULT_END = _today.strftime("%Y-%m-%dT00")
DEFAULT_START_DATE = _three_months_ago.strftime("%Y-%m-%d")
DEFAULT_END_DATE = _today.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1) Demand & Forecast  (electricity/rto/region-data)
# ---------------------------------------------------------------------------
def fetch_demand_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly demand and day-ahead forecast for all BAs in one batch."""
    if respondents is None:
        respondents = MAJOR_BA

    df = _fetch_eia_paginated(
        api_key=api_key,
        url="https://api.eia.gov/v2/electricity/rto/region-data/data/",
        facets={"respondent": respondents, "type": ["D", "DF"]},
        start=start,
        end=end,
    )

    if df.empty:
        return df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df


# ---------------------------------------------------------------------------
# 2) Interchange  (electricity/rto/interchange-data)
# ---------------------------------------------------------------------------
def fetch_interchange_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly interchange for all BAs in one batch."""
    if respondents is None:
        respondents = MAJOR_BA

    df = _fetch_eia_paginated(
        api_key=api_key,
        url="https://api.eia.gov/v2/electricity/rto/interchange-data/data/",
        facets={"fromba": respondents},
        start=start,
        end=end,
    )

    if df.empty:
        return df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df


# ---------------------------------------------------------------------------
# 3) Generation by Fuel Type  (electricity/rto/fuel-type-data)
# ---------------------------------------------------------------------------
def fetch_fuel_type_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly generation by fuel type for all BAs in one batch."""
    if respondents is None:
        respondents = MAJOR_BA

    df = _fetch_eia_paginated(
        api_key=api_key,
        url="https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/",
        facets={"respondent": respondents},
        start=start,
        end=end,
    )

    if df.empty:
        return df

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df


# ---------------------------------------------------------------------------
# 4) Natural Gas Price  (natural-gas/pri/fut)
# ---------------------------------------------------------------------------
def fetch_natural_gas_prices(
    api_key: str,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Fetch Henry Hub natural gas spot prices (daily)."""
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RNGWHHD",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != HTTP_OK:
        return pd.DataFrame()

    data = response.json().get("response", {}).get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df.rename(columns={"value": "ng_price", "period": "date"})


# ---------------------------------------------------------------------------
# 5) Weather  (Open-Meteo Archive API)
# ---------------------------------------------------------------------------
def fetch_weather_data(
    ba_coords: dict[str, tuple[float, float]] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Fetch daily temperature data for each BA's representative city."""
    if ba_coords is None:
        ba_coords = BA_WEATHER_COORDS

    all_frames = []
    for ba_code, (lat, lon) in ba_coords.items():
        df = _fetch_single_weather(lat, lon, start_date, end_date)
        if not df.empty:
            df["ba"] = ba_code
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def _fetch_single_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "America/New_York",
    }
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != HTTP_OK:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                return pd.DataFrame()
            data = response.json().get("daily", {})
            if not data or "time" not in data:
                return pd.DataFrame()
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(data["time"]),
                    "max_temp": pd.to_numeric(data.get("temperature_2m_max"), errors="coerce"),
                    "min_temp": pd.to_numeric(data.get("temperature_2m_min"), errors="coerce"),
                }
            )
            df["avg_temp"] = (df["max_temp"] + df["min_temp"]) / 2
            return df
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            print(f"     Weather attempt {attempt + 1} failed: {type(e).__name__}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal: paginated EIA API fetcher (batches all facet values in one call)
# ---------------------------------------------------------------------------
def _fetch_eia_paginated(
    api_key: str,
    url: str,
    facets: dict[str, list[str]],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch data from an EIA API endpoint with pagination.

    All facet values are sent in a single request, avoiding the
    overhead of one HTTP call per BA. Errors are logged verbosely
    so silent failures (HTTP 403/429, empty responses, schema changes)
    surface in CI logs.
    """
    max_length = 5000
    params: dict = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": max_length,
    }

    for facet_name, facet_values in facets.items():
        params[f"facets[{facet_name}][]"] = facet_values

    all_data: list[dict] = []
    offset = 0
    endpoint = url.split("/v2/")[-1] if "/v2/" in url else url

    while True:
        params["offset"] = offset

        try:
            response = requests.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"  ❌ {endpoint} request failed at offset {offset}: {e}")
            break

        if response.status_code != HTTP_OK:
            print(f"  ❌ {endpoint} HTTP {response.status_code} at offset {offset}")
            print(f"     Response body: {response.text[:300]}")
            break

        try:
            payload = response.json().get("response", {})
        except ValueError as e:
            print(f"  ❌ {endpoint} JSON parse failed: {e}")
            break

        records = payload.get("data", [])

        if not records:
            if offset == 0:
                print(f"  ⚠ {endpoint} returned 0 records on first page")
                print(f"     Facets: {facets}")
                print(f"     Date range: {start} → {end}")
            break

        all_data.extend(records)

        if len(records) < max_length:
            break

        offset += max_length

    print(f"  → {endpoint}: {len(all_data)} total records fetched")

    if not all_data:
        return pd.DataFrame()

    return pd.DataFrame(all_data)
