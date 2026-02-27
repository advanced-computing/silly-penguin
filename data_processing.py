import pandas as pd


def merge_daily_demand_weather(
    daily_demand: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    daily_demand = daily_demand.copy()
    daily_demand["date"] = pd.to_datetime(daily_demand["period"].dt.date)

    demand_only = daily_demand[daily_demand["type-name"] == "Demand"]

    avg_demand = demand_only.groupby("date")["value"].mean().reset_index()
    avg_demand.rename(columns={"value": "avg_demand_mwh"})

    merged_df = avg_demand.merge(weather_df, on="date", how="inner")

    return merged_df


def calculate_grid_kpis(df_pivot):
    """
    Extracts the latest Actual Demand, Forecast, and calculates the Delta.
    Returns (last_actual, last_forecast, delta) or (None, None, None) if data is missing.
    """
    if df_pivot.empty:
        return None, None, None

    try:
        last_actual = df_pivot["Demand"].iloc[0]
        last_forecast = df_pivot["Day-ahead demand forecast"].iloc[0]
        delta = last_actual - last_forecast
        return last_actual, last_forecast, delta
    except (KeyError, IndexError):
        return None, None, None
