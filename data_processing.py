import pandas as pd


def merge_daily_demand_weather(
    eia_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Filters EIA data to actual demand, resamples to daily average, and merges with weather."""
    if eia_df.empty or weather_df.empty:
        return pd.DataFrame()

    # Filter for Actual Demand only
    demand_only = eia_df[eia_df["type-name"] == "Demand"].copy()

    # Resample to daily average
    daily_demand = demand_only.resample("D", on="period")["value"].mean().reset_index()
    daily_demand.rename(
        columns={"period": "date", "value": "avg_demand_mwh"}, inplace=True
    )

    # Merge with weather data
    merged_df = pd.merge(daily_demand, weather_df, on="date", how="inner")

    return merged_df
