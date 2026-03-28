"""Data processing for the Grid Intelligence Platform.

Modules:
1. Anomaly Detection — forecast error monitoring and alerting
2. Arbitrage Signals — cross-market interchange pattern identification
3. Renewable Siting Score — investment opportunity scoring per BA
4. Compliance Reports — automated regulatory data summaries
5. Core utilities — pivoting, KPIs, geographic helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
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

ALERT_WINDOW_HOURS = 6
PERCENTILE_RED = 95
PERCENTILE_YELLOW = 90

PEAK_HOURS_START = 16
PEAK_HOURS_END = 19

MIN_HOURS_FOR_GROWTH = 48  # at least 2 days of hourly data

RENEWABLE_FUELS = ["SUN", "WND", "WAT"]


# ===================================================================
# 1) ANOMALY DETECTION
# ===================================================================
def compute_forecast_errors(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly forecast errors for all BAs."""
    if demand_df.empty:
        return pd.DataFrame()

    pivot = demand_df.pivot_table(
        index=["period", "respondent"], columns="type-name", values="value"
    ).reset_index()

    if "Demand" not in pivot.columns or "Day-ahead demand forecast" not in pivot.columns:
        return pd.DataFrame()

    pivot["error"] = pivot["Demand"] - pivot["Day-ahead demand forecast"]
    pivot["abs_error"] = pivot["error"].abs()
    demand_safe = pivot["Demand"].replace(0, np.nan)
    pivot["ape"] = (pivot["abs_error"] / demand_safe) * 100
    return pivot.sort_values(["respondent", "period"])


def detect_anomalies(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Detect BAs with sustained high forecast errors.

    For each BA:
    - Compute historical error distribution (full dataset)
    - Check if the last N hours are consistently above the 95th/90th percentile
    - Return alert status: RED / YELLOW / NORMAL
    """
    errors = compute_forecast_errors(demand_df)
    if errors.empty:
        return pd.DataFrame()

    alerts = []
    for ba in errors["respondent"].unique():
        ba_data = errors[errors["respondent"] == ba].copy()
        if len(ba_data) < ALERT_WINDOW_HOURS:
            continue

        p95 = ba_data["abs_error"].quantile(PERCENTILE_RED / 100)
        p90 = ba_data["abs_error"].quantile(PERCENTILE_YELLOW / 100)

        recent = ba_data.tail(ALERT_WINDOW_HOURS)
        recent_mean = recent["abs_error"].mean()
        recent_max = recent["abs_error"].max()
        hours_above_p95 = (recent["abs_error"] > p95).sum()
        hours_above_p90 = (recent["abs_error"] > p90).sum()

        if hours_above_p95 >= ALERT_WINDOW_HOURS // 2:
            status = "RED"
        elif hours_above_p90 >= ALERT_WINDOW_HOURS // 2:
            status = "YELLOW"
        else:
            status = "NORMAL"

        alerts.append(
            {
                "ba": ba,
                "status": status,
                "recent_mean_error": recent_mean,
                "recent_max_error": recent_max,
                "p95_threshold": p95,
                "p90_threshold": p90,
                "hours_above_p95": int(hours_above_p95),
                "hours_above_p90": int(hours_above_p90),
                "latest_error": recent["abs_error"].iloc[-1],
                "latest_demand": recent["Demand"].iloc[-1]
                if "Demand" in recent.columns
                else np.nan,
            }
        )

    result = pd.DataFrame(alerts)
    status_order = {"RED": 0, "YELLOW": 1, "NORMAL": 2}
    result["_sort"] = result["status"].map(status_order)
    return result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)


def get_ba_error_distribution(demand_df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Get full error time series for a single BA (for distribution charts)."""
    errors = compute_forecast_errors(demand_df)
    if errors.empty:
        return pd.DataFrame()
    return errors[errors["respondent"] == ba].copy()


# ===================================================================
# 2) ARBITRAGE SIGNALS
# ===================================================================
def compute_interchange_patterns(interchange_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly interchange patterns for all BA pairs.

    Returns average flow by (fromba, toba, hour) with volatility metrics.
    """
    if interchange_df.empty:
        return pd.DataFrame()

    work = interchange_df.copy()
    work["hour"] = work["period"].dt.hour

    patterns = (
        work.groupby(["fromba", "toba", "hour"])["value"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "avg_flow",
                "std": "volatility",
                "count": "n_obs",
                "min": "min_flow",
                "max": "max_flow",
            }
        )
    )
    patterns["volatility"] = patterns["volatility"].fillna(0)
    return patterns


def identify_arbitrage_opportunities(interchange_df: pd.DataFrame) -> pd.DataFrame:
    """Identify persistent directional flow patterns (arbitrage signals).

    Looks for BA pairs where flow is consistently in one direction
    during peak hours (16:00-19:00), suggesting price differentials.
    """
    patterns = compute_interchange_patterns(interchange_df)
    if patterns.empty:
        return pd.DataFrame()

    peak = patterns[(patterns["hour"] >= PEAK_HOURS_START) & (patterns["hour"] <= PEAK_HOURS_END)]

    if peak.empty:
        return pd.DataFrame()

    signals = (
        peak.groupby(["fromba", "toba"])
        .agg(
            peak_avg_flow=("avg_flow", "mean"),
            peak_volatility=("volatility", "mean"),
            peak_min=("min_flow", "min"),
            peak_max=("max_flow", "max"),
            total_obs=("n_obs", "sum"),
        )
        .reset_index()
    )

    # Signal strength: high flow + low volatility = strong signal
    flow_abs = signals["peak_avg_flow"].abs()
    max_flow = flow_abs.max()
    if max_flow > 0:
        signals["directional_strength"] = flow_abs / max_flow

    max_vol = signals["peak_volatility"].max()
    if max_vol > 0:
        signals["consistency"] = 1 - (signals["peak_volatility"] / max_vol)
    else:
        signals["consistency"] = 1.0

    signals["signal_score"] = (
        signals.get("directional_strength", 0) * 0.6 + signals.get("consistency", 0) * 0.4
    ) * 100

    signals["direction"] = np.where(signals["peak_avg_flow"] > 0, "Export →", "← Import")

    return signals.sort_values("signal_score", ascending=False).reset_index(drop=True)


def get_pair_hourly_profile(interchange_df: pd.DataFrame, fromba: str, toba: str) -> pd.DataFrame:
    """Get 24-hour flow profile for a specific BA pair."""
    patterns = compute_interchange_patterns(interchange_df)
    if patterns.empty:
        return pd.DataFrame()

    pair = patterns[(patterns["fromba"] == fromba) & (patterns["toba"] == toba)].copy()
    return pair.sort_values("hour")


# ===================================================================
# 3) RENEWABLE SITING SCORE
# ===================================================================
def _score_demand_growth(demand_df: pd.DataFrame, ba: str) -> float:
    """Score demand growth trend for a BA (0-100).

    Compares the latest week's average demand to the earliest week's,
    then scales the percentage change. Even small changes get amplified
    because 3 months is too short for secular growth trends — we're
    really measuring seasonal demand trajectory.
    """
    if demand_df.empty:
        return 50.0
    ba_d = demand_df[(demand_df["respondent"] == ba) & (demand_df["type-name"] == "Demand")]
    if len(ba_d) < MIN_HOURS_FOR_GROWTH:
        return 50.0

    ba_d = ba_d.sort_values("period").copy()
    ba_d["date"] = ba_d["period"].dt.date
    daily = ba_d.groupby("date")["value"].mean()

    n = len(daily)
    week = min(7, max(n // 4, 1))
    early = daily.iloc[:week].mean()
    late = daily.iloc[-week:].mean()

    if early <= 0:
        return 50.0

    growth_pct = (late - early) / early * 100
    # Scale: even 0.5% growth → noticeable score shift. Cap 0–100.
    return min(max(50 + growth_pct * 25, 0), 100)


def _score_renewable_headroom(fuel_df: pd.DataFrame, ba: str) -> tuple[float, float]:
    """Score renewable headroom (low share = high score). Returns (score, current_pct)."""
    if fuel_df.empty or "fueltype" not in fuel_df.columns:
        return 50.0, 0.0
    ba_f = fuel_df[fuel_df["respondent"] == ba]
    if ba_f.empty:
        return 50.0, 0.0
    total_gen = ba_f["value"].sum()
    if total_gen <= 0:
        return 50.0, 0.0
    renew_gen = ba_f[ba_f["fueltype"].isin(RENEWABLE_FUELS)]["value"].sum()
    pct = (renew_gen / total_gen) * 100
    return max(100 - pct, 0), pct


def _score_import_dependence(interchange_df: pd.DataFrame, ba: str) -> float:
    """Score import dependence (net importer = high score)."""
    if interchange_df.empty or "fromba" not in interchange_df.columns:
        return 50.0
    ba_i = interchange_df[interchange_df["fromba"] == ba]
    if ba_i.empty:
        return 50.0
    net_avg = ba_i["value"].mean()
    max_abs = interchange_df["value"].abs().quantile(0.95)
    if max_abs <= 0:
        return 50.0
    return min(max(50 - (net_avg / max_abs) * 50, 0), 100)


def _score_fossil_transition(fuel_df: pd.DataFrame, ba: str) -> float:
    """Score fossil dependence (high fossil = high transition opportunity)."""
    if fuel_df.empty or "fueltype" not in fuel_df.columns:
        return 50.0
    ba_f = fuel_df[fuel_df["respondent"] == ba]
    if ba_f.empty:
        return 50.0
    total_gen = ba_f["value"].sum()
    if total_gen <= 0:
        return 50.0
    fossil_types = ["NG", "COL", "OIL"]
    fossil_gen = ba_f[ba_f["fueltype"].isin(fossil_types)]["value"].sum()
    return min((fossil_gen / total_gen) * 100, 100)


def compute_renewable_siting_scores(
    demand_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute renewable energy investment opportunity score per BA.

    Score components (0-100 each, weighted 25% each):
    - Demand growth trend: rising demand = more opportunity
    - Renewable headroom: low current share = more room to grow
    - Import dependence: net importer = needs local generation
    - Fossil transition: heavy fossil = transition opportunity
    """
    scores = []
    for ba_code, meta in BA_METADATA.items():
        demand_score = _score_demand_growth(demand_df, ba_code)
        renewable_score, current_pct = _score_renewable_headroom(fuel_df, ba_code)
        import_score = _score_import_dependence(interchange_df, ba_code)
        fossil_score = _score_fossil_transition(fuel_df, ba_code)

        composite = (
            demand_score * 0.25 + renewable_score * 0.25 + import_score * 0.25 + fossil_score * 0.25
        )

        scores.append(
            {
                "ba": ba_code,
                "name": meta["name"],
                "demand_growth_score": demand_score,
                "renewable_headroom_score": renewable_score,
                "current_renewable_pct": current_pct,
                "import_dependence_score": import_score,
                "fossil_transition_score": fossil_score,
                "composite_score": composite,
            }
        )

    return (
        pd.DataFrame(scores).sort_values("composite_score", ascending=False).reset_index(drop=True)
    )


# ===================================================================
# 4) COMPLIANCE REPORTS
# ===================================================================
def _compliance_demand_section(demand_df: pd.DataFrame, ba: str) -> dict | None:
    """Build demand section of compliance report."""
    if demand_df.empty:
        return None
    ba_demand = demand_df[(demand_df["respondent"] == ba) & (demand_df["type-name"] == "Demand")]
    if ba_demand.empty:
        return None
    return {
        "period_start": str(ba_demand["period"].min()),
        "period_end": str(ba_demand["period"].max()),
        "avg_demand_mwh": round(ba_demand["value"].mean(), 1),
        "peak_demand_mwh": round(ba_demand["value"].max(), 1),
        "min_demand_mwh": round(ba_demand["value"].min(), 1),
        "total_hours": len(ba_demand),
    }


def _compliance_forecast_section(demand_df: pd.DataFrame, ba: str) -> dict | None:
    """Build forecast accuracy section of compliance report."""
    errors = compute_forecast_errors(demand_df)
    ba_errors = errors[errors["respondent"] == ba] if not errors.empty else pd.DataFrame()
    if ba_errors.empty:
        return None
    return {
        "mape": round(ba_errors["ape"].mean(), 2),
        "mae_mwh": round(ba_errors["abs_error"].mean(), 1),
        "max_error_mwh": round(ba_errors["abs_error"].max(), 1),
        "bias_mwh": round(ba_errors["error"].mean(), 1),
    }


def _compliance_interchange_section(interchange_df: pd.DataFrame, ba: str) -> dict | None:
    """Build interchange section of compliance report."""
    if interchange_df.empty or "fromba" not in interchange_df.columns:
        return None
    ba_int = interchange_df[interchange_df["fromba"] == ba]
    if ba_int.empty:
        return None
    return {
        "avg_net_mwh": round(ba_int["value"].mean(), 1),
        "total_net_mwh": round(ba_int["value"].sum(), 1),
        "peak_export_mwh": round(ba_int["value"].max(), 1),
        "peak_import_mwh": round(ba_int["value"].min(), 1),
        "n_trading_partners": ba_int["toba"].nunique(),
    }


def _compliance_genmix_section(fuel_df: pd.DataFrame, ba: str) -> dict | None:
    """Build generation mix section of compliance report."""
    if fuel_df.empty or "fueltype" not in fuel_df.columns:
        return None
    ba_fuel = fuel_df[fuel_df["respondent"] == ba]
    if ba_fuel.empty:
        return None
    total = ba_fuel["value"].sum()
    mix = {}
    if total > 0:
        for ft in ba_fuel["fueltype"].unique():
            ft_val = ba_fuel[ba_fuel["fueltype"] == ft]["value"].sum()
            mix[ft] = round((ft_val / total) * 100, 1)
    return {
        "total_generation_mwh": round(total, 0),
        "fuel_shares_pct": mix,
    }


def generate_compliance_summary(
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    ba: str,
) -> dict:
    """Generate a FERC-style compliance data summary for a BA."""
    sections: dict = {}

    demand_sec = _compliance_demand_section(demand_df, ba)
    if demand_sec:
        sections["demand"] = demand_sec

    forecast_sec = _compliance_forecast_section(demand_df, ba)
    if forecast_sec:
        sections["forecast_accuracy"] = forecast_sec

    int_sec = _compliance_interchange_section(interchange_df, ba)
    if int_sec:
        sections["interchange"] = int_sec

    genmix_sec = _compliance_genmix_section(fuel_df, ba)
    if genmix_sec:
        sections["generation_mix"] = genmix_sec

    return {
        "ba": ba,
        "ba_name": BA_METADATA.get(ba, {}).get("name", ba),
        "sections": sections,
    }


# ===================================================================
# 5) CORE UTILITIES (kept from original)
# ===================================================================
def prepare_demand_pivot(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Pivot demand data into columns with error metrics."""
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


def compute_generation_mix(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Compute daily generation by fuel type."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if ba is not None:
        work = work[work["respondent"] == ba]
    if work.empty or "fueltype" not in work.columns:
        return pd.DataFrame()

    work["date"] = pd.to_datetime(work["period"].dt.date)
    daily_mix = work.groupby(["date", "fueltype"])["value"].mean().reset_index()
    return daily_mix.rename(columns={"value": "avg_generation_mwh"})


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


def compute_net_interchange(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
    """Compute net interchange for a BA over time."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if ba is not None:
        work = work[work["fromba"] == ba]
    if work.empty:
        return pd.DataFrame()

    net = work.groupby("period")["value"].sum().reset_index()
    net = net.rename(columns={"value": "net_interchange_mwh"})
    return net.sort_values("period")


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
    return daily_demand.merge(weather_work[weather_cols], on="date", how="inner")


def build_geographic_summary(
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build summary with one row per BA for map visualization."""
    rows = []
    for ba_code, meta in BA_METADATA.items():
        row: dict = {
            "ba": ba_code,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
        }

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

        if not interchange_df.empty and "fromba" in interchange_df.columns:
            ba_int = interchange_df[interchange_df["fromba"] == ba_code]
            if not ba_int.empty:
                row["avg_net_interchange"] = ba_int["value"].mean()

        if not fuel_df.empty and "fueltype" in fuel_df.columns:
            ba_fuel = fuel_df[fuel_df["respondent"] == ba_code]
            if not ba_fuel.empty:
                total_gen = ba_fuel["value"].sum()
                renewable_gen = ba_fuel[ba_fuel["fueltype"].isin(RENEWABLE_FUELS)]["value"].sum()
                if total_gen > 0:
                    row["renewable_share"] = (renewable_gen / total_gen) * 100

        rows.append(row)

    return pd.DataFrame(rows)
