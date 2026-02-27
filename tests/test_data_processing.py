from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to import path (so CI can import data_processing.py from repo root)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import calculate_grid_kpis, merge_daily_demand_weather  # noqa: E402

# Avoid "magic values" warnings by defining constants
DEMAND_1 = 100.0
DEMAND_2 = 200.0
FORECAST_IGNORED = 150.0
DEMAND_DAY2 = 300.0

ACTUAL_1 = 5000.0
ACTUAL_2 = 4900.0
FORECAST_1 = 4800.0
FORECAST_2 = 4950.0
DELTA_1 = 200.0


def test_merge_daily_demand_weather():
    eia_data = pd.DataFrame(
        {
            "period": pd.to_datetime(
                [
                    "2025-01-01 10:00:00",
                    "2025-01-01 11:00:00",
                    "2025-01-01 10:00:00",
                    "2025-01-02 10:00:00",
                ]
            ),
            "type-name": ["Demand", "Demand", "Day-ahead demand forecast", "Demand"],
            "value": [DEMAND_1, DEMAND_2, FORECAST_IGNORED, DEMAND_DAY2],
        }
    )

    weather_data = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "avg_temp": [15.5, 16.0]}
    )

    expected_output = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "avg_demand_mwh": [(DEMAND_1 + DEMAND_2) / 2, DEMAND_DAY2],
            "avg_temp": [15.5, 16.0],
        }
    )

    result = merge_daily_demand_weather(eia_data, weather_data)

    pd.testing.assert_frame_equal(result, expected_output)


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

    # 2. Execution
    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    # 3. Assertion
    assert last_actual is None
    assert last_forecast is None
    assert delta is None
