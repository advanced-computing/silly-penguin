"""Data processing, transformation, and metric computation."""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 1) Demand Forecast Processing
# ---------------------------------------------------------------------------
def prepare_demand_pivot(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Pivot demand data into columns: Demand, Day-ahead demand forecast, plus error metrics."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if ba is not None:
        work = work[work["respondent"] == ba]

    if work.empty:
        return pd.DataFrame()

    pivot = work.pivot_table(index="period", columns="type-name", values="value")
    pivot = pivot.sort_index(ascending=False)

    if {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
        pivot["Forecast Error"] = pivot["Demand"] - pivot["Day-ahead demand forecast"]
        pivot["Absolute Error"] = pivot["Forecast Error"].abs()
        demand_safe = pivot["Demand"].replace(0, pd.NA)
        pivot["APE"] = (pivot["Absolute Error"] / demand_safe) * 100
        pivot["Error 24h MA"] = pivot["Forecast Error"].rolling(24).mean()

    return pivot


def compute_daily_demand_stats(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Aggregate hourly demand data into daily statistics per BA."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if ba is not None:
        work = work[work["respondent"] == ba]

    pivot = work.pivot_table(index="period", columns="type-name", values="value")
    pivot = pivot.sort_index()

    if not {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
        return pd.DataFrame()

    pivot["error"] = pivot["Demand"] - pivot["Day-ahead demand forecast"]
    pivot["abs_error"] = pivot["error"].abs()
    demand_safe = pivot["Demand"].replace(0, pd.NA)
    pivot["ape"] = (pivot["abs_error"] / demand_safe) * 100
    pivot["date"] = pd.to_datetime(pivot.index.date)

    daily = (
        pivot.groupby("date")
        .agg(
            avg_demand_mwh=("Demand", "mean"),
            avg_forecast_mwh=("Day-ahead demand forecast", "mean"),
            avg_error_mwh=("error", "mean"),
            mape=("ape", "mean"),
            peak_demand_mwh=("Demand", "max"),
        )
        .reset_index()
    )
    return daily


def compute_ba_mape_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MAPE ranking across all balancing authorities."""
    if df.empty:
        return pd.DataFrame()

    results = []
    for ba in df["respondent"].unique():
        ba_data = df[df["respondent"] == ba]
        pivot = ba_data.pivot_table(index="period", columns="type-name", values="value")

        if not {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
            continue

        demand_safe = pivot["Demand"].replace(0, pd.NA)
        ape = ((pivot["Demand"] - pivot["Day-ahead demand forecast"]).abs() / demand_safe) * 100
        results.append(
            {
                "ba": ba,
                "mape": ape.mean(),
                "mae": (pivot["Demand"] - pivot["Day-ahead demand forecast"]).abs().mean(),
                "avg_demand": pivot["Demand"].mean(),
                "n_hours": len(pivot),
            }
        )

    if not results:
        return pd.DataFrame()

    ranking = pd.DataFrame(results).sort_values("mape", ascending=True).reset_index(drop=True)
    return ranking


def compute_error_heatmap_data(df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Create hour-of-day vs date heatmap data for forecast errors."""
    if df.empty:
        return pd.DataFrame()

    work = df[df["respondent"] == ba].copy()
    pivot = work.pivot_table(index="period", columns="type-name", values="value")

    if not {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
        return pd.DataFrame()

    pivot["error"] = pivot["Demand"] - pivot["Day-ahead demand forecast"]
    pivot["hour"] = pivot.index.hour
    pivot["date"] = pd.to_datetime(pivot.index.date)

    heatmap = pivot.pivot_table(index="date", columns="hour", values="error", aggfunc="mean")
    return heatmap


# ---------------------------------------------------------------------------
# 2) Interchange Processing
# ---------------------------------------------------------------------------
def compute_net_interchange(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Compute net interchange (imports - exports) for a BA over time."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    if ba is not None:
        work = work[work["fromba"] == ba]

    if work.empty:
        return pd.DataFrame()

    net = work.groupby("period")["value"].sum().reset_index()
    net = net.rename(columns={"value": "net_interchange_mwh"})
    net = net.sort_values("period")
    return net


def compute_daily_interchange(df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Aggregate hourly interchange to daily level for a specific BA."""
    if df.empty:
        return pd.DataFrame()

    work = df[df["fromba"] == ba].copy()
    if work.empty:
        return pd.DataFrame()

    work["date"] = pd.to_datetime(work["period"].dt.date)
    daily = (
        work.groupby("date")["value"]
        .agg(["mean", "sum", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "avg_interchange",
                "sum": "total_interchange",
                "min": "min_interchange",
                "max": "max_interchange",
            }
        )
    )
    return daily


def compute_hourly_interchange_pattern(df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Compute average interchange by hour-of-day for a BA."""
    if df.empty:
        return pd.DataFrame()

    work = df[df["fromba"] == ba].copy()
    if work.empty:
        return pd.DataFrame()

    work["hour"] = work["period"].dt.hour
    pattern = work.groupby("hour")["value"].mean().reset_index()
    pattern = pattern.rename(columns={"value": "avg_interchange_mwh"})
    return pattern


# ---------------------------------------------------------------------------
# 3) Fuel Type / Generation Mix Processing
# ---------------------------------------------------------------------------
def compute_generation_mix(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Compute daily generation by fuel type for stacked area charts."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if ba is not None:
        work = work[work["respondent"] == ba]

    if work.empty or "fueltype" not in work.columns:
        return pd.DataFrame()

    work["date"] = pd.to_datetime(work["period"].dt.date)
    daily_mix = work.groupby(["date", "fueltype"])["value"].mean().reset_index()
    daily_mix = daily_mix.rename(columns={"value": "avg_generation_mwh"})
    return daily_mix


def compute_fuel_share(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Compute percentage share of each fuel type over time."""
    mix = compute_generation_mix(df, ba)
    if mix.empty:
        return pd.DataFrame()

    daily_total = mix.groupby("date")["avg_generation_mwh"].sum().reset_index()
    daily_total = daily_total.rename(columns={"avg_generation_mwh": "total_generation"})

    merged = mix.merge(daily_total, on="date")
    total_safe = merged["total_generation"].replace(0, pd.NA)
    merged["share_pct"] = (merged["avg_generation_mwh"] / total_safe) * 100
    return merged


def compute_ng_price_vs_generation(
    fuel_df: pd.DataFrame,
    price_df: pd.DataFrame,
    ba: str | None = None,
) -> pd.DataFrame:
    """Merge natural gas prices with generation data for price-response analysis."""
    if fuel_df.empty or price_df.empty:
        return pd.DataFrame()

    mix = compute_fuel_share(fuel_df, ba)
    if mix.empty:
        return pd.DataFrame()

    price_clean = price_df[["date", "ng_price"]].copy()
    price_clean["date"] = pd.to_datetime(price_clean["date"].dt.date)

    merged = mix.merge(price_clean, on="date", how="inner")
    return merged


# ---------------------------------------------------------------------------
# 4) Weather Integration
# ---------------------------------------------------------------------------
def merge_demand_weather(
    daily_demand: pd.DataFrame,
    weather_df: pd.DataFrame,
    ba: str | None = None,
) -> pd.DataFrame:
    """Merge daily demand stats with weather data."""
    if daily_demand.empty or weather_df.empty:
        return pd.DataFrame()

    weather_work = weather_df.copy()
    if ba is not None:
        weather_work = weather_work[weather_work["ba"] == ba]

    if weather_work.empty:
        return pd.DataFrame()

    weather_cols = ["date", "avg_temp", "max_temp", "min_temp"]
    merged = daily_demand.merge(weather_work[weather_cols], on="date", how="inner")
    return merged


def label_weather_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify days into cold / mild / hot using quantiles."""
    if df.empty or "avg_temp" not in df.columns:
        return df

    result = df.copy()
    low_q = result["avg_temp"].quantile(0.1)
    high_q = result["avg_temp"].quantile(0.9)

    conditions = [
        result["avg_temp"] <= low_q,
        result["avg_temp"] >= high_q,
    ]
    choices = ["Cold Extreme", "Hot Extreme"]
    result["weather_regime"] = pd.Series(
        pd.Categorical(
            ["Mild / Normal"] * len(result),
            categories=["Cold Extreme", "Mild / Normal", "Hot Extreme"],
        )
    )

    for condition, label in zip(conditions, choices, strict=True):
        result.loc[condition, "weather_regime"] = label

    return result


# ---------------------------------------------------------------------------
# 5) KPI Helpers
# ---------------------------------------------------------------------------
def calculate_grid_kpis(
    df_pivot: pd.DataFrame,
) -> tuple[float | None, float | None, float | None]:
    """Extract latest Actual Demand, Forecast, and Delta."""
    if df_pivot.empty:
        return None, None, None

    try:
        last_actual = df_pivot["Demand"].iloc[0]
        last_forecast = df_pivot["Day-ahead demand forecast"].iloc[0]
        delta = last_actual - last_forecast
    except (KeyError, IndexError):
        return None, None, None
    else:
        return last_actual, last_forecast, delta


# ---------------------------------------------------------------------------
# 6) Geographic Helpers
# ---------------------------------------------------------------------------
BA_METADATA: dict[str, dict] = {
    "CISO": {"name": "California ISO", "lat": 36.78, "lon": -119.42},
    "ERCO": {"name": "ERCOT (Texas)", "lat": 31.97, "lon": -99.90},
    "PJM": {"name": "PJM Interconnection", "lat": 39.95, "lon": -75.16},
    "MISO": {"name": "Midcontinent ISO", "lat": 41.88, "lon": -87.63},
    "NYIS": {"name": "New York ISO", "lat": 40.71, "lon": -74.01},
    "ISNE": {"name": "ISO New England", "lat": 42.36, "lon": -71.06},
    "SWPP": {"name": "Southwest Power Pool", "lat": 35.47, "lon": -97.52},
    "SOCO": {"name": "Southern Company", "lat": 33.75, "lon": -84.39},
    "TVA": {"name": "Tennessee Valley Authority", "lat": 36.16, "lon": -86.78},
    "BPAT": {"name": "Bonneville Power Admin", "lat": 45.52, "lon": -122.68},
}


def build_geographic_summary(
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a summary dataframe with one row per BA for map visualization."""
    rows = []
    for ba_code, meta in BA_METADATA.items():
        row: dict = {
            "ba": ba_code,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
        }

        # Demand MAPE
        if not demand_df.empty:
            ba_demand = demand_df[demand_df["respondent"] == ba_code]
            pivot = ba_demand.pivot_table(index="period", columns="type-name", values="value")
            if {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
                demand_safe = pivot["Demand"].replace(0, pd.NA)
                ape = (
                    (pivot["Demand"] - pivot["Day-ahead demand forecast"]).abs() / demand_safe
                ) * 100
                row["mape"] = ape.mean()
                row["avg_demand"] = pivot["Demand"].mean()

        # Net interchange
        if not interchange_df.empty and "fromba" in interchange_df.columns:
            ba_int = interchange_df[interchange_df["fromba"] == ba_code]
            if not ba_int.empty:
                row["avg_net_interchange"] = ba_int["value"].mean()

        # Renewable share
        if not fuel_df.empty and "fueltype" in fuel_df.columns:
            ba_fuel = fuel_df[fuel_df["respondent"] == ba_code]
            if not ba_fuel.empty:
                total_gen = ba_fuel["value"].sum()
                renewable_types = ["SUN", "WND", "WAT"]
                renewable_gen = ba_fuel[ba_fuel["fueltype"].isin(renewable_types)]["value"].sum()
                if total_gen > 0:
                    row["renewable_share"] = (renewable_gen / total_gen) * 100

        rows.append(row)

    return pd.DataFrame(rows)
