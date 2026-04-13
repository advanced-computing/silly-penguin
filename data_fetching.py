"""Data fetching for all grid intelligence data sources.

Sources:
  1. EIA v2 API — demand, interchange, fuel mix, natural gas prices
  2. Open-Meteo — temperature per BA
  3. gridstatus library — ISO LMP (PJM, CAISO, ERCOT)
  4. NREL Developer API — solar & wind resource quality
  5. LBNL interconnection queue — local Excel file
"""

from __future__ import annotations

import os

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

HTTP_OK = 200
MAX_EIA_PAGE_LENGTH = 5000
WEATHER_MAX_RETRIES = 3
LAST_WEATHER_RETRY_INDEX = WEATHER_MAX_RETRIES - 1
WEATHER_RETRY_BASE_SECONDS = 5
NREL_REQUEST_TIMEOUT = 20
OPEN_METEO_TIMEOUT = 60
EIA_REQUEST_TIMEOUT = 60
NG_PRICE_TIMEOUT = 30
NREL_PACING_SECONDS = 0.3
EXCEL_DATE_ORIGIN = "1899-12-30"
NREL_EMPTY_WIND_RESULT = {"wind_speed": None, "wind_power": None}

# ---------------------------------------------------------------------------
# BAs we track + region metadata
# ---------------------------------------------------------------------------
MAJOR_BA = [
    "CISO",
    "ERCO",
    "PJM",
    "MISO",
    "NYIS",
    "ISNE",
    "SWPP",
    "SOCO",
    "TVA",
    "BPAT",
]

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

FUEL_TYPES = ["NG", "SUN", "WND", "NUC", "COL", "WAT", "OIL", "OTH"]

# ISOs we pull LMP for (gridstatus). PJM is auto-prepended at runtime
# if PJM_API_KEY is set in the environment.
ISO_LMP_SOURCES = ["CAISO", "ERCOT"]

# Mapping from our ISO label → gridstatus class name. ERCOT is Pascal case
# in gridstatus, while the others are all-caps.
ISO_CLASS_MAP = {
    "PJM": "PJM",
    "CAISO": "CAISO",
    "ERCOT": "Ercot",
    "MISO": "MISO",
    "NYISO": "NYISO",
    "ISONE": "ISONE",
    "SPP": "SPP",
}

# Mapping from our BA codes to gridstatus ISO names
BA_TO_ISO = {
    "PJM": "PJM",
    "CISO": "CAISO",
    "ERCO": "ERCOT",
    "MISO": "MISO",
    "NYIS": "NYISO",
    "ISNE": "ISONE",
    "SWPP": "SPP",
}

ISO_CLASS_MAP = {
    "PJM": "PJM",
    "CAISO": "CAISO",
    "ERCOT": "Ercot",  # ← Pascal case, not all-caps
    "MISO": "MISO",
    "NYISO": "NYISO",
    "ISONE": "ISONE",
    "SPP": "SPP",
}

# LBNL region → US state codes, used to filter queue data precisely to each BA
# West and Southeast in LBNL are non-ISO; we use state-level filtering
BA_STATES: dict[str, list[str]] = {
    "CISO": ["CA"],
    "ERCO": ["TX"],
    "PJM": [
        "PA",
        "NJ",
        "MD",
        "DE",
        "VA",
        "WV",
        "OH",
        "KY",
        "IL",
        "IN",
        "MI",
        "NC",
        "DC",
    ],
    "MISO": ["MN", "WI", "IA", "IL", "IN", "MI", "MO", "AR", "LA", "MS", "ND", "SD"],
    "NYIS": ["NY"],
    "ISNE": ["MA", "CT", "RI", "VT", "NH", "ME"],
    "SWPP": ["KS", "OK", "NE", "ND", "SD", "NM", "AR", "TX"],
    "SOCO": ["GA", "AL", "FL", "MS"],
    "TVA": ["TN", "AL", "MS", "KY", "VA", "NC", "GA"],
    "BPAT": ["OR", "WA", "ID", "MT", "WY", "CA", "NV", "UT"],
}

# NREL — a broader set of sampling points for siting analysis.
# State centroids + key metro areas. Keeps count manageable (<100).
NREL_SAMPLE_POINTS: list[tuple[str, str, float, float]] = [
    ("CA", "Fresno (Central Valley)", 36.75, -119.78),
    ("CA", "Mojave", 35.05, -118.17),
    ("CA", "Los Angeles", 34.05, -118.25),
    ("TX", "West Texas (Midland)", 31.99, -102.08),
    ("TX", "Houston", 29.76, -95.37),
    ("TX", "Dallas", 32.78, -96.80),
    ("TX", "Panhandle (Amarillo)", 35.22, -101.83),
    ("NY", "Buffalo", 42.89, -78.88),
    ("NY", "Albany", 42.65, -73.76),
    ("PA", "Philadelphia", 39.95, -75.17),
    ("PA", "Pittsburgh", 40.44, -79.99),
    ("OH", "Columbus", 39.96, -82.99),
    ("IL", "Chicago", 41.88, -87.63),
    ("MI", "Detroit", 42.33, -83.05),
    ("VA", "Richmond", 37.54, -77.44),
    ("NC", "Raleigh", 35.78, -78.64),
    ("GA", "Atlanta", 33.75, -84.39),
    ("FL", "Orlando", 28.54, -81.38),
    ("FL", "Miami", 25.76, -80.19),
    ("AL", "Birmingham", 33.52, -86.80),
    ("TN", "Nashville", 36.16, -86.78),
    ("TN", "Memphis", 35.15, -90.05),
    ("KY", "Louisville", 38.25, -85.76),
    ("MS", "Jackson", 32.30, -90.18),
    ("LA", "New Orleans", 29.95, -90.07),
    ("AR", "Little Rock", 34.74, -92.29),
    ("OK", "Oklahoma City", 35.47, -97.52),
    ("KS", "Wichita", 37.69, -97.34),
    ("NE", "Omaha", 41.26, -95.93),
    ("IA", "Des Moines", 41.59, -93.62),
    ("MN", "Minneapolis", 44.98, -93.27),
    ("WI", "Milwaukee", 43.04, -87.91),
    ("MO", "St. Louis", 38.63, -90.20),
    ("ND", "Bismarck", 46.81, -100.78),
    ("SD", "Pierre", 44.37, -100.35),
    ("WA", "Seattle", 47.61, -122.33),
    ("OR", "Portland", 45.52, -122.68),
    ("ID", "Boise", 43.62, -116.21),
    ("MT", "Billings", 45.78, -108.50),
    ("WY", "Cheyenne", 41.14, -104.82),
    ("CO", "Denver", 39.74, -104.99),
    ("UT", "Salt Lake City", 40.76, -111.89),
    ("NV", "Las Vegas", 36.17, -115.14),
    ("NM", "Albuquerque", 35.08, -106.65),
    ("AZ", "Phoenix", 33.45, -112.07),
    ("MA", "Boston", 42.36, -71.06),
    ("CT", "Hartford", 41.76, -72.68),
    ("ME", "Portland", 43.66, -70.26),
]

# Dynamic time range
_today = datetime.now()
_three_months_ago = _today - timedelta(days=90)
_thirty_days_ago = _today - timedelta(days=30)
DEFAULT_START = _three_months_ago.strftime("%Y-%m-%dT00")
DEFAULT_END = _today.strftime("%Y-%m-%dT00")
DEFAULT_START_DATE = _three_months_ago.strftime("%Y-%m-%d")
DEFAULT_END_DATE = _today.strftime("%Y-%m-%d")
LMP_START_DATE = _thirty_days_ago.strftime("%Y-%m-%d")
LMP_END_DATE = _today.strftime("%Y-%m-%d")


# ===========================================================================
# 1) EIA: Demand & Forecast
# ===========================================================================
def fetch_demand_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly demand + day-ahead forecast for all BAs."""
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


# ===========================================================================
# 2) EIA: Interchange
# ===========================================================================
def fetch_interchange_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly inter-BA interchange for all BAs."""
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


# ===========================================================================
# 3) EIA: Generation by Fuel Type
# ===========================================================================
def fetch_fuel_type_data(
    api_key: str,
    respondents: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """Fetch hourly generation by fuel type for all BAs."""
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


# ===========================================================================
# 4) EIA: Natural Gas Price (Henry Hub)
# ===========================================================================
def fetch_natural_gas_prices(
    api_key: str,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Fetch daily Henry Hub NG spot price."""
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
        "length": MAX_EIA_PAGE_LENGTH,
    }
    try:
        response = requests.get(url, params=params, timeout=NG_PRICE_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        print(f"  ❌ NG price request failed: {exc}")
        return pd.DataFrame()

    if response.status_code != HTTP_OK:
        print(f"  ❌ NG price HTTP {response.status_code}")
        return pd.DataFrame()

    data = response.json().get("response", {}).get("data", [])
    if not data:
        print("  ⚠ NG price returned 0 records")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df.rename(columns={"value": "ng_price", "period": "date"})


# ===========================================================================
# 5) Weather (Open-Meteo)
# ===========================================================================
def fetch_weather_data(
    ba_coords: dict[str, tuple[float, float]] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Fetch daily temperature for each BA's representative city."""
    if ba_coords is None:
        ba_coords = BA_WEATHER_COORDS

    all_frames = []
    for ba_code, (lat, lon) in ba_coords.items():
        df = _fetch_single_weather(lat, lon, start_date, end_date)
        if not df.empty:
            df["ba"] = ba_code
            all_frames.append(df)

    if not all_frames:
        print("  ⚠ Weather: all locations failed")
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _fetch_single_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch weather for a single location with retries."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "America/New_York",
    }

    for attempt in range(WEATHER_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT)
            if response.status_code != HTTP_OK:
                if attempt < LAST_WEATHER_RETRY_INDEX:
                    time.sleep(WEATHER_RETRY_BASE_SECONDS * (attempt + 1))
                    continue
                return pd.DataFrame()

            data = response.json().get("daily", {})
            if not data or "time" not in data:
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(data["time"]),
                    "max_temp": pd.to_numeric(
                        data.get("temperature_2m_max"),
                        errors="coerce",
                    ),
                    "min_temp": pd.to_numeric(
                        data.get("temperature_2m_min"),
                        errors="coerce",
                    ),
                }
            )
            df["avg_temp"] = (df["max_temp"] + df["min_temp"]) / 2
            return df
        except requests.exceptions.RequestException:
            if attempt < LAST_WEATHER_RETRY_INDEX:
                time.sleep(WEATHER_RETRY_BASE_SECONDS * (attempt + 1))

    return pd.DataFrame()


# ===========================================================================
# 6) ISO LMP via gridstatus (PJM, CAISO, ERCOT)
# ===========================================================================
def fetch_iso_lmp(
    isos: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly day-ahead LMP for selected ISOs via gridstatus.

    Returns a unified DataFrame with columns:
      iso, time, location, location_type, market, lmp, energy, congestion, loss
    """
    # Late import: only needed if this function is actually called
    import gridstatus
    from gridstatus import Markets

    # Default ISO list: skip PJM unless API key is set
    if isos is None:
        isos = ["CAISO", "ERCOT"]
        if os.getenv("PJM_API_KEY"):
            isos.insert(0, "PJM")
        else:
            print("  ℹ Skipping PJM: PJM_API_KEY not set in environment.")

    # Default date range: last 30 days (kept inline to remain self-contained)
    today = pd.Timestamp.now().normalize()
    start = pd.Timestamp(start_date) if start_date else today - pd.Timedelta(days=30)
    end = pd.Timestamp(end_date) if end_date else today

    all_frames: list[pd.DataFrame] = []
    for iso_name in isos:
        print(f"  Fetching {iso_name} LMP from {start.date()} to {end.date()}...")
        try:
            class_name = ISO_CLASS_MAP.get(iso_name, iso_name)
            iso_class = getattr(gridstatus, class_name)

            # PJM accepts api_key kwarg; others don't
            if iso_name == "PJM":
                iso = iso_class(api_key=os.getenv("PJM_API_KEY"))
            else:
                iso = iso_class()

            lmp = iso.get_lmp(
                date=start,
                end=end,
                market=Markets.DAY_AHEAD_HOURLY,
                locations="ALL",
                verbose=False,
            )
            if lmp is None or lmp.empty:
                print(f"    ⚠ {iso_name} returned empty")
                continue

            lmp = lmp.copy()
            lmp["iso"] = iso_name

            # --- FIX: handle duplicate time columns BEFORE rename ---
            # Some ISOs (notably CAISO) return both "Time" and "Interval Start".
            # Drop "Time" if "Interval Start" is present so we don't end up
            # with two columns named "time" after rename.
            if "Interval Start" in lmp.columns and "Time" in lmp.columns:
                lmp = lmp.drop(columns=["Time"])

            rename_map = {
                "Time": "time",
                "Interval Start": "time",
                "Location": "location",
                "Location Type": "location_type",
                "Market": "market",
                "LMP": "lmp",
                "Energy": "energy",
                "Congestion": "congestion",
                "Loss": "loss",
            }
            lmp = lmp.rename(
                columns={k: v for k, v in rename_map.items() if k in lmp.columns}
            )

            # Defensive: if duplicates somehow survived, keep first
            if lmp.columns.duplicated().any():
                lmp = lmp.loc[:, ~lmp.columns.duplicated()]

            keep = [
                c
                for c in [
                    "iso",
                    "time",
                    "location",
                    "location_type",
                    "market",
                    "lmp",
                    "energy",
                    "congestion",
                    "loss",
                ]
                if c in lmp.columns
            ]
            lmp = lmp[keep]

            # Filter to zone/hub locations only (skip thousands of nodes)
            if "location_type" in lmp.columns:
                mask = (
                    lmp["location_type"]
                    .astype(str)
                    .str.contains(
                        "ZONE|HUB|AGGREGATE|TRADING_HUB",
                        case=False,
                        na=False,
                    )
                )
                if mask.any():
                    lmp = lmp[mask]

            lmp["time"] = pd.to_datetime(lmp["time"], utc=True, errors="coerce")
            lmp = lmp.dropna(subset=["time"])
            for col in ["lmp", "energy", "congestion", "loss"]:
                if col in lmp.columns:
                    lmp[col] = pd.to_numeric(lmp[col], errors="coerce")

            print(
                f"    → {iso_name}: {len(lmp)} hourly LMP records "
                f"({lmp['location'].nunique()} locations)"
            )
            all_frames.append(lmp)
        except Exception as exc:
            print(f"    ❌ {iso_name} failed: {type(exc).__name__}: {exc}")
            continue

    if not all_frames:
        print("  ⚠ No LMP data fetched from any ISO")
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


# ===========================================================================
# 7) NREL Resource Quality (Solar + Wind)
# ===========================================================================
def fetch_nrel_resources(api_key: str) -> pd.DataFrame:
    """Fetch annual solar and wind resource metrics for sample points."""
    if not api_key:
        print("  ⚠ NREL_API_KEY not set, skipping resource data")
        return pd.DataFrame()

    rows = []
    for state, label, lat, lon in NREL_SAMPLE_POINTS:
        solar = _fetch_nrel_solar(api_key, lat, lon)
        wind = _fetch_nrel_wind(api_key, lat, lon)
        row = {
            "state": state,
            "label": label,
            "lat": lat,
            "lon": lon,
            "solar_ghi_annual_kwh_m2": solar.get("ghi"),
            "solar_dni_annual_kwh_m2": solar.get("dni"),
            "wind_speed_100m_avg": wind.get("wind_speed"),
            "wind_power_100m_avg": wind.get("wind_power"),
        }
        rows.append(row)
        time.sleep(NREL_PACING_SECONDS)

    df = pd.DataFrame(rows)
    print(
        f"  → NREL: {len(df)} sample locations "
        f"({df['solar_ghi_annual_kwh_m2'].notna().sum()} with solar, "
        f"{df['wind_speed_100m_avg'].notna().sum()} with wind)"
    )
    return df


def _fetch_nrel_solar(api_key: str, lat: float, lon: float) -> dict[str, float | None]:
    """Fetch solar resource annual averages via NREL Solar Resource API."""
    url = "https://developer.nrel.gov/api/solar/solar_resource/v1.json"
    params = {"api_key": api_key, "lat": lat, "lon": lon}

    try:
        response = requests.get(url, params=params, timeout=NREL_REQUEST_TIMEOUT)
        if response.status_code != HTTP_OK:
            return {}

        outputs = response.json().get("outputs", {})
        ghi = outputs.get("avg_ghi", {}).get("annual")
        dni = outputs.get("avg_dni", {}).get("annual")
        return {
            "ghi": float(ghi) if ghi else None,
            "dni": float(dni) if dni else None,
        }
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return {}


def _fetch_nrel_wind(api_key: str, lat: float, lon: float) -> dict[str, float | None]:
    """Fetch wind resource via NREL Wind Toolkit summary API."""
    url = "https://developer.nrel.gov/api/wind-toolkit/v2/wind/site-count.json"
    params = {"api_key": api_key, "lat": lat, "lon": lon}
    result: dict[str, float | None] = {}

    try:
        response = requests.get(url, params=params, timeout=NREL_REQUEST_TIMEOUT)
        if response.status_code == HTTP_OK:
            result = NREL_EMPTY_WIND_RESULT.copy()
    except requests.exceptions.RequestException:
        result = {}

    return result


# ===========================================================================
# 8) LBNL Interconnection Queue (local Excel)
# ===========================================================================
def load_lbnl_queue(
    excel_path: str = "data/LBNL_Ix_Queue_Data_File_thru2024_v2.xlsx",
) -> pd.DataFrame:
    """Load LBNL interconnection queue data from local Excel.

    Returns a cleaned DataFrame with one row per project, with BA assignment.
    """
    path = Path(excel_path)
    if not path.exists():
        print(f"  ⚠ LBNL file not found at {path}")
        return pd.DataFrame()

    print(f"  Reading {path}...")
    df = pd.read_excel(path, sheet_name="03. Complete Queue Data", header=1)
    print(f"  Raw LBNL rows: {len(df)}")

    keep_cols = [
        "q_id",
        "q_status",
        "q_date",
        "prop_date",
        "on_date",
        "wd_date",
        "ia_date",
        "IA_status_clean",
        "county",
        "state",
        "poi_name",
        "region",
        "project_name",
        "utility",
        "developer",
        "project_type",
        "type_clean",
        "mw1",
        "q_year",
        "prop_year",
    ]
    df = df[[column for column in keep_cols if column in df.columns]].copy()

    if "type_clean" in df.columns:
        df["type_clean"] = (
            df["type_clean"]
            .astype(str)
            .str.replace("\u00ac\u2020", " ", regex=False)
            .str.strip()
        )

    df["mw1"] = pd.to_numeric(df["mw1"], errors="coerce")
    df["q_year"] = pd.to_numeric(df["q_year"], errors="coerce")
    df["ba"] = df.apply(_assign_ba_from_state_region, axis=1)

    for date_column in ["q_date", "prop_date", "on_date", "wd_date", "ia_date"]:
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(
                df[date_column],
                errors="coerce",
                unit="D",
                origin=EXCEL_DATE_ORIGIN,
            )

    print(
        f"  → LBNL queue: {len(df)} projects ({df['ba'].notna().sum()} matched to a BA)"
    )
    return df


def _assign_ba_from_state_region(row: pd.Series) -> str | None:
    """Pick the most appropriate BA for a queue project.

    Priority:
      1. Direct ISO match (PJM, CAISO, MISO, ERCOT, NYISO, ISO-NE, SPP)
      2. State-based match for non-ISO regions (West, Southeast)
    """
    region = str(row.get("region", "")).strip()
    state = str(row.get("state", "")).strip().upper()

    region_to_ba = {
        "PJM": "PJM",
        "CAISO": "CISO",
        "MISO": "MISO",
        "ERCOT": "ERCO",
        "NYISO": "NYIS",
        "ISO-NE": "ISNE",
        "SPP": "SWPP",
    }
    if region in region_to_ba:
        return region_to_ba[region]

    if state == "CA":
        return "CISO"
    if state == "TX":
        return "ERCO"
    if state in ["OR", "WA", "ID"]:
        return "BPAT"
    if state in ["TN", "KY"]:
        return "TVA"
    if state in ["GA", "AL", "FL", "MS"]:
        if state in ["GA", "FL"]:
            return "SOCO"
        return "SOCO"
    return None


# ===========================================================================
# Internal: EIA paginated fetcher with verbose error logging
# ===========================================================================
def _fetch_eia_paginated(  # noqa: PLR0911
    api_key: str,
    url: str,
    facets: dict[str, list[str]],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch from EIA with pagination and surface all failures."""
    params: dict[str, Any] = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": MAX_EIA_PAGE_LENGTH,
    }
    for facet_name, facet_values in facets.items():
        params[f"facets[{facet_name}][]"] = facet_values

    all_data: list[dict[str, Any]] = []
    offset = 0
    endpoint = url.split("/v2/")[-1] if "/v2/" in url else url

    while True:
        params["offset"] = offset
        try:
            response = requests.get(url, params=params, timeout=EIA_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            print(f"  ❌ {endpoint} request failed at offset {offset}: {exc}")
            break

        if response.status_code != HTTP_OK:
            print(f"  ❌ {endpoint} HTTP {response.status_code} at offset {offset}")
            print(f"     Response body: {response.text[:300]}")
            break

        try:
            payload = response.json().get("response", {})
        except ValueError as exc:
            print(f"  ❌ {endpoint} JSON parse failed: {exc}")
            break

        records = payload.get("data", [])
        if not records:
            if offset == 0:
                print(f"  ⚠ {endpoint} returned 0 records on first page")
                print(f"     Facets: {facets}")
                print(f"     Date range: {start} → {end}")
            break

        all_data.extend(records)
        if len(records) < MAX_EIA_PAGE_LENGTH:
            break
        offset += MAX_EIA_PAGE_LENGTH

    print(f"  → {endpoint}: {len(all_data)} total records fetched")
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data)
