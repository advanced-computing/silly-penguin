from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing import calculate_grid_kpis  # noqa: E402

ACTUAL_1 = 5000.0
ACTUAL_2 = 4900.0
FORECAST_1 = 4800.0
FORECAST_2 = 4950.0
DELTA_1 = 200.0


def test_calculate_grid_kpis_happy_path() -> None:
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


def test_calculate_grid_kpis_missing_column() -> None:
    df_pivot = pd.DataFrame({"Demand": [ACTUAL_1, ACTUAL_2]})

    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    assert last_actual is None
    assert last_forecast is None
    assert delta is None


def test_calculate_grid_kpis_empty_df() -> None:
    df_pivot = pd.DataFrame()

    last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

    assert last_actual is None
    assert last_forecast is None
    assert delta is None
