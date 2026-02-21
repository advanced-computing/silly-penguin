import pandas as pd
import pytest
from data_processing import merge_daily_demand_weather
from data_processing import calculate_grid_kpis


def test_merge_daily_demand_weather():
    # Mock Inputs
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
            "value": [100.0, 200.0, 150.0, 300.0],
        }
    )

    weather_data = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "avg_temp": [15.5, 16.0]}
    )

    # Expected Output
    # Jan 1 Demand should be average of 100 and 200 = 150. Forecast is ignored.
    expected_output = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "avg_demand_mwh": [150.0, 300.0],
            "avg_temp": [15.5, 16.0],
        }
    )

    # Execution
    result = merge_daily_demand_weather(eia_data, weather_data)

    # Assertion
    pd.testing.assert_frame_equal(result, expected_output)


def test_calculate_grid_kpis_happy_path():
    """Test Case 1: Standard valid data with both columns."""
    # 1. Setup Mock Input
    df_pivot = pd.DataFrame(
        {"Demand": [5000.0, 4900.0], "Day-ahead demand forecast": [4800.0, 4950.0]}
    )

    # 2. Execution
    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    # 3. Assertion
    assert last_actual == 5000.0
    assert last_forecast == 4800.0
    assert delta == 200.0


def test_calculate_grid_kpis_missing_column():
    """Test Case 2: Edge Case - Missing 'Day-ahead demand forecast' column."""
    # 1. Setup Mock Input (API failure simulation)
    df_pivot = pd.DataFrame({"Demand": [5000.0, 4900.0]})

    # 2. Execution
    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    # 3. Assertion
    assert last_actual is None
    assert last_forecast is None
    assert delta is None


def test_calculate_grid_kpis_empty_df():
    """Test Case 3: Edge Case - Completely empty DataFrame."""
    # 1. Setup Mock Input
    df_pivot = pd.DataFrame()

    # 2. Execution
    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    # 3. Assertion
    assert last_actual is None
    assert last_forecast is None
    assert delta is None
