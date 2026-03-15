from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to import path (so CI can import data_processing.py from repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import calculate_grid_kpis, merge_demand_weather  # noqa: E402

# Avoid "magic values" warnings by defining constants
DEMAND_1 = 100.0
DEMAND_2 = 200.0
DEMAND_DAY2 = 300.0

ACTUAL_1 = 5000.0
ACTUAL_2 = 4900.0
FORECAST_1 = 4800.0
FORECAST_2 = 4950.0
DELTA_1 = 200.0

TEMP_1 = 15.5
TEMP_2 = 16.0
MAX_TEMP_1 = 20.0
MIN_TEMP_1 = 11.0
EXPECTED_MERGED_ROWS = 2
MAX_TEMP_2 = 21.0
MIN_TEMP_2 = 11.0


def test_merge_demand_weather():
    """Test merging daily demand stats with weather data."""
    daily_demand = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "avg_demand_mwh": [(DEMAND_1 + DEMAND_2) / 2, DEMAND_DAY2],
            "avg_forecast_mwh": [DEMAND_1, DEMAND_2],
            "avg_error_mwh": [50.0, 100.0],
            "mape": [5.0, 10.0],
            "peak_demand_mwh": [DEMAND_2, DEMAND_DAY2],
        }
    )

    weather_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "avg_temp": [TEMP_1, TEMP_2],
            "max_temp": [MAX_TEMP_1, MAX_TEMP_2],
            "min_temp": [MIN_TEMP_1, MIN_TEMP_2],
            "ba": ["CISO", "CISO"],
        }
    )

    result = merge_demand_weather(daily_demand, weather_data, ba="CISO")

    assert len(result) == EXPECTED_MERGED_ROWS
    assert "avg_demand_mwh" in result.columns
    assert "avg_temp" in result.columns
    assert result["avg_temp"].iloc[0] == TEMP_1


def test_merge_demand_weather_empty_input():
    """Test that empty inputs return empty DataFrame."""
    empty_df = pd.DataFrame()
    weather_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "avg_temp": [TEMP_1],
            "max_temp": [MAX_TEMP_1],
            "min_temp": [MIN_TEMP_1],
            "ba": ["CISO"],
        }
    )

    result = merge_demand_weather(empty_df, weather_data)
    assert result.empty


def test_merge_demand_weather_no_matching_ba():
    """Test that non-matching BA returns empty DataFrame."""
    daily_demand = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "avg_demand_mwh": [DEMAND_1],
        }
    )
    weather_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"]),
            "avg_temp": [TEMP_1],
            "max_temp": [MAX_TEMP_1],
            "min_temp": [MIN_TEMP_1],
            "ba": ["ERCO"],
        }
    )

    result = merge_demand_weather(daily_demand, weather_data, ba="CISO")
    assert result.empty


def test_calculate_grid_kpis_happy_path():
    df_pivot = pd.DataFrame(
        {
            "Demand": [ACTUAL_1, ACTUAL_2],
            "Day-ahead demand forecast": [FORECAST_1, FORECAST_2],
        }
    )

    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    assert last_actual == ACTUAL_1
    assert last_forecast == FORECAST_1
    assert delta == DELTA_1


def test_calculate_grid_kpis_missing_column():
    df_pivot = pd.DataFrame({"Demand": [ACTUAL_1, ACTUAL_2]})

    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    assert last_actual is None
    assert last_forecast is None
    assert delta is None


def test_calculate_grid_kpis_empty_df():
    df_pivot = pd.DataFrame()

    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    assert last_actual is None
    assert last_forecast is None
    assert delta is None
