import pandas as pd
import pytest
from data_processing import merge_daily_demand_weather


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
