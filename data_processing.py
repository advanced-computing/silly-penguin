"""Data processing for the Grid Intelligence Platform.

Modules (aligned with app.py sections):
  1. Anomaly Detection   — forecast error + LMP price spike monitoring
  2. Arbitrage Signals   — inter-BA macro flows + ISO zonal LMP micro spreads
  3. Transition Scoring  — renewable investment opportunity (queue + resource)
  4. Compliance Reports  — FERC-style summaries + cross-module signal injection
  5. Executive Briefing  — consolidated per-BA snapshot combining all modules
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BA_METADATA: dict[str, dict] = {
    "CISO": {"name": "California ISO", "lat": 36.78, "lon": -119.42, "iso": "CAISO"},
    "ERCO": {"name": "ERCOT (Texas)", "lat": 31.97, "lon": -99.90, "iso": "ERCOT"},
    "PJM": {"name": "PJM Interconnection", "lat": 39.95, "lon": -75.16, "iso": "PJM"},
    "MISO": {"name": "Midcontinent ISO", "lat": 41.88, "lon": -87.63, "iso": "MISO"},
    "NYIS": {"name": "New York ISO", "lat": 40.71, "lon": -74.01, "iso": "NYISO"},
    "ISNE": {"name": "ISO New England", "lat": 42.36, "lon": -71.06, "iso": "ISONE"},
    "SWPP": {"name": "Southwest Power Pool", "lat": 35.47, "lon": -97.52, "iso": "SPP"},
    "SOCO": {"name": "Southern Company", "lat": 33.75, "lon": -84.39, "iso": None},
    "TVA": {"name": "Tennessee Valley Authority", "lat": 36.16, "lon": -86.78, "iso": None},
    "BPAT": {"name": "Bonneville Power Admin", "lat": 45.52, "lon": -122.68, "iso": None},
}

ALERT_WINDOW_HOURS = 6
PERCENTILE_RED = 95
PERCENTILE_YELLOW = 90
MIN_HOURS_FOR_ALERT = 3

NERC_PEAK_START = 14
NERC_PEAK_END = 19

RENEWABLE_FUELS = ["SUN", "WND", "WAT"]
FOSSIL_FUELS = ["NG", "COL", "OIL"]

MIN_HOURS_FOR_GROWTH = 48
MIN_ISO_LOCATIONS_FOR_SPREAD = 2
MIN_HOURS_FOR_SPREAD = 24

LMP_SPIKE_MULTIPLIER = 3.0
LMP_NEGATIVE_THRESHOLD = -10.0


# ===================================================================
# 1) ANOMALY DETECTION (EIA forecast errors)
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
    """Flag BAs with sustained high forecast errors."""
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

        if hours_above_p95 >= MIN_HOURS_FOR_ALERT:
            status = "RED"
        elif hours_above_p90 >= MIN_HOURS_FOR_ALERT:
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
    if result.empty:
        return result
    status_order = {"RED": 0, "YELLOW": 1, "NORMAL": 2}
    result["_sort"] = result["status"].map(status_order)
    return result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)


def get_ba_error_distribution(demand_df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Get full error time series for a single BA."""
    errors = compute_forecast_errors(demand_df)
    if errors.empty:
        return pd.DataFrame()
    return errors[errors["respondent"] == ba].copy()


# ===================================================================
# 1b) LMP ANOMALY DETECTION (NEW)
# ===================================================================
def detect_lmp_anomalies(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """Detect LMP price spikes and negative-price events per ISO location."""
    if lmp_df.empty or "lmp" not in lmp_df.columns:
        return pd.DataFrame()

    df = lmp_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["iso", "location", "time"])

    alerts = []
    for (iso, location), group in df.groupby(["iso", "location"]):
        if len(group) < ALERT_WINDOW_HOURS:
            continue

        historical_median = group["lmp"].median()
        if not np.isfinite(historical_median) or historical_median <= 0:
            continue

        recent = group.tail(ALERT_WINDOW_HOURS)
        recent_avg = recent["lmp"].mean()
        recent_min = recent["lmp"].min()
        recent_max = recent["lmp"].max()

        is_spike = recent_avg > historical_median * LMP_SPIKE_MULTIPLIER
        is_negative = recent_min < LMP_NEGATIVE_THRESHOLD

        if is_spike:
            status = "SPIKE"
        elif is_negative:
            status = "NEGATIVE"
        else:
            status = "NORMAL"

        alerts.append(
            {
                "iso": iso,
                "location": location,
                "status": status,
                "recent_avg_lmp": recent_avg,
                "recent_max_lmp": recent_max,
                "recent_min_lmp": recent_min,
                "historical_median": historical_median,
                "spike_ratio": recent_avg / historical_median if historical_median > 0 else None,
                "latest_lmp": recent["lmp"].iloc[-1],
                "latest_time": recent["time"].iloc[-1],
            }
        )

    result = pd.DataFrame(alerts)
    if result.empty:
        return result
    status_order = {"SPIKE": 0, "NEGATIVE": 1, "NORMAL": 2}
    result["_sort"] = result["status"].map(status_order)
    return (
        result.sort_values(["_sort", "spike_ratio"], ascending=[True, False])
        .drop(columns=["_sort"])
        .reset_index(drop=True)
    )


def get_lmp_time_series(lmp_df: pd.DataFrame, iso: str, location: str) -> pd.DataFrame:
    """Get full LMP time series for a single location."""
    if lmp_df.empty:
        return pd.DataFrame()
    return (
        lmp_df[(lmp_df["iso"] == iso) & (lmp_df["location"] == location)].sort_values("time").copy()
    )


# ===================================================================
# 2) ARBITRAGE SIGNALS
# ===================================================================
def compute_interchange_patterns(interchange_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly interchange patterns for all BA pairs."""
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
    """Identify persistent directional flow patterns during peak hours."""
    patterns = compute_interchange_patterns(interchange_df)
    if patterns.empty:
        return pd.DataFrame()

    peak = patterns[(patterns["hour"] >= NERC_PEAK_START) & (patterns["hour"] <= NERC_PEAK_END)]
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

    flow_abs = signals["peak_avg_flow"].abs()
    max_flow = flow_abs.max()
    signals["directional_strength"] = flow_abs / max_flow if max_flow > 0 else 0

    max_vol = signals["peak_volatility"].max()
    if max_vol > 0:
        signals["consistency"] = 1 - (signals["peak_volatility"] / max_vol)
    else:
        signals["consistency"] = 1.0

    signals["signal_score"] = (
        signals["directional_strength"] * 0.6 + signals["consistency"] * 0.4
    ) * 100
    signals["direction"] = np.where(signals["peak_avg_flow"] > 0, "Export →", "← Import")
    return signals.sort_values("signal_score", ascending=False).reset_index(drop=True)


def get_pair_hourly_profile(interchange_df: pd.DataFrame, fromba: str, toba: str) -> pd.DataFrame:
    """24-hour flow profile for a specific BA pair."""
    patterns = compute_interchange_patterns(interchange_df)
    if patterns.empty:
        return pd.DataFrame()
    pair = patterns[(patterns["fromba"] == fromba) & (patterns["toba"] == toba)].copy()
    return pair.sort_values("hour")


def compute_lmp_zonal_spreads(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute price spreads between zones within each ISO (NEW)."""
    if lmp_df.empty or "lmp" not in lmp_df.columns:
        return pd.DataFrame()

    df = lmp_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour

    rows = []
    for iso, iso_group in df.groupby("iso"):
        if iso_group["location"].nunique() < MIN_ISO_LOCATIONS_FOR_SPREAD:
            continue

        wide = iso_group.pivot_table(index="time", columns="location", values="lmp", aggfunc="mean")
        if wide.shape[1] < MIN_ISO_LOCATIONS_FOR_SPREAD:
            continue

        peak_mask = (wide.index.hour >= NERC_PEAK_START) & (wide.index.hour <= NERC_PEAK_END)

        loc_list = list(wide.columns)
        for i, loc_a in enumerate(loc_list):
            for loc_b in loc_list[i + 1 :]:
                spread = (wide[loc_a] - wide[loc_b]).dropna()
                if len(spread) < MIN_HOURS_FOR_SPREAD:
                    continue
                peak_spread_series = spread[peak_mask[: len(spread)]]
                rows.append(
                    {
                        "iso": iso,
                        "zone_a": loc_a,
                        "zone_b": loc_b,
                        "mean_spread": spread.abs().mean(),
                        "peak_spread": peak_spread_series.abs().mean()
                        if len(peak_spread_series) > 0
                        else 0,
                        "spread_volatility": spread.std(),
                        "n_hours": len(spread),
                    }
                )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    max_peak = result["peak_spread"].max()
    max_vol = result["spread_volatility"].max()
    strength = (result["peak_spread"] / max_peak) if max_peak > 0 else 0
    consistency = (1 - result["spread_volatility"] / max_vol) if max_vol > 0 else 1.0
    result["signal_score"] = (strength * 0.6 + consistency * 0.4) * 100
    return result.sort_values("signal_score", ascending=False).reset_index(drop=True)


# ===================================================================
# 3) TRANSITION SCORING
# ===================================================================
def _score_demand_growth(demand_df: pd.DataFrame, ba: str) -> float:
    if demand_df.empty:
        return 50.0
    ba_d = demand_df[(demand_df["respondent"] == ba) & (demand_df["type-name"] == "Demand")]
    if len(ba_d) < MIN_HOURS_FOR_GROWTH:
        return 50.0

    ba_d = ba_d.sort_values("period").copy()
    ba_d["date"] = ba_d["period"].dt.date
    daily = ba_d.groupby("date")["value"].mean()

    n = len(daily)
    third = max(n // 3, 1)
    early = daily.iloc[:third].mean()
    late = daily.iloc[-third:].mean()
    if early <= 0:
        return 50.0
    growth_pct = (late - early) / early * 100
    return min(max(50 + growth_pct * 10, 0), 100)


def _score_renewable_headroom(fuel_df: pd.DataFrame, ba: str) -> tuple[float, float]:
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
    if fuel_df.empty or "fueltype" not in fuel_df.columns:
        return 50.0
    ba_f = fuel_df[fuel_df["respondent"] == ba]
    if ba_f.empty:
        return 50.0
    total_gen = ba_f["value"].sum()
    if total_gen <= 0:
        return 50.0
    fossil_gen = ba_f[ba_f["fueltype"].isin(FOSSIL_FUELS)]["value"].sum()
    return min((fossil_gen / total_gen) * 100, 100)


def _score_queue_activity(queue_summary: pd.DataFrame, ba: str) -> dict:
    if queue_summary.empty:
        return {
            "queue_active_score": 50.0,
            "queue_completion_score": 50.0,
            "active_projects": 0,
            "active_capacity_mw": 0,
            "completion_rate": None,
        }
    ba_row = queue_summary[queue_summary["ba"] == ba]
    if ba_row.empty:
        return {
            "queue_active_score": 25.0,
            "queue_completion_score": 50.0,
            "active_projects": 0,
            "active_capacity_mw": 0,
            "completion_rate": None,
        }

    row = ba_row.iloc[0]
    active_projects = int(row.get("active_projects", 0) or 0)
    active_mw = float(row.get("active_capacity_mw", 0) or 0)
    completion_rate = row.get("completion_rate")

    max_active_mw = queue_summary["active_capacity_mw"].max()
    active_score = (active_mw / max_active_mw) * 100 if max_active_mw > 0 else 50.0

    if completion_rate is None or pd.isna(completion_rate):
        completion_score = 50.0
    else:
        completion_score = float(completion_rate) * 100

    return {
        "queue_active_score": min(active_score, 100),
        "queue_completion_score": min(completion_score, 100),
        "active_projects": active_projects,
        "active_capacity_mw": active_mw,
        "completion_rate": completion_rate,
    }


def compute_transition_scores(
    demand_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    queue_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Composite renewable transition opportunity score per BA."""
    scores = []
    for ba_code, meta in BA_METADATA.items():
        demand_score = _score_demand_growth(demand_df, ba_code)
        renewable_score, current_pct = _score_renewable_headroom(fuel_df, ba_code)
        import_score = _score_import_dependence(interchange_df, ba_code)
        fossil_score = _score_fossil_transition(fuel_df, ba_code)

        if queue_summary is not None and not queue_summary.empty:
            q = _score_queue_activity(queue_summary, ba_code)
            queue_active = q["queue_active_score"]
            queue_completion = q["queue_completion_score"]
            active_projects = q["active_projects"]
            active_capacity_mw = q["active_capacity_mw"]
            completion_rate = q["completion_rate"]
            composite = (
                demand_score * 0.15
                + renewable_score * 0.15
                + import_score * 0.15
                + fossil_score * 0.15
                + queue_active * 0.20
                + queue_completion * 0.20
            )
        else:
            queue_active = np.nan
            queue_completion = np.nan
            active_projects = 0
            active_capacity_mw = 0
            completion_rate = None
            composite = (
                demand_score * 0.25
                + renewable_score * 0.25
                + import_score * 0.25
                + fossil_score * 0.25
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
                "queue_active_score": queue_active,
                "queue_completion_score": queue_completion,
                "active_projects": active_projects,
                "active_capacity_mw": active_capacity_mw,
                "completion_rate": completion_rate,
                "composite_score": composite,
            }
        )

    return (
        pd.DataFrame(scores).sort_values("composite_score", ascending=False).reset_index(drop=True)
    )


def get_queue_breakdown_for_ba(queue_type_summary: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Resource-type breakdown of queue for a BA."""
    if queue_type_summary.empty:
        return pd.DataFrame()
    ba_q = queue_type_summary[queue_type_summary["ba"] == ba].copy()
    return ba_q.sort_values("active_mw", ascending=False)


# ===================================================================
# 4) COMPLIANCE REPORTS
# ===================================================================
def _compliance_demand_section(demand_df: pd.DataFrame, ba: str) -> dict | None:
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


def _compliance_cross_module_section(
    ba: str,
    anomaly_alerts: pd.DataFrame | None,
    transition_scores: pd.DataFrame | None,
) -> dict | None:
    """NEW: inject signals from other modules into compliance report."""
    out: dict = {}
    if anomaly_alerts is not None and not anomaly_alerts.empty:
        ba_alert = anomaly_alerts[anomaly_alerts["ba"] == ba]
        if not ba_alert.empty:
            row = ba_alert.iloc[0]
            out["anomaly_status"] = row["status"]
            out["hours_above_p95"] = int(row["hours_above_p95"])
    if transition_scores is not None and not transition_scores.empty:
        ba_score = transition_scores[transition_scores["ba"] == ba]
        if not ba_score.empty:
            row = ba_score.iloc[0]
            out["transition_composite_score"] = round(float(row["composite_score"]), 1)
            out["active_queue_mw"] = float(row.get("active_capacity_mw", 0) or 0)
    return out if out else None


def generate_compliance_summary(  # noqa: PLR0913
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    ba: str,
    anomaly_alerts: pd.DataFrame | None = None,
    transition_scores: pd.DataFrame | None = None,
) -> dict:
    """FERC-style compliance summary with cross-module injection."""
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

    cross_sec = _compliance_cross_module_section(ba, anomaly_alerts, transition_scores)
    if cross_sec:
        sections["cross_module_signals"] = cross_sec

    return {
        "ba": ba,
        "ba_name": BA_METADATA.get(ba, {}).get("name", ba),
        "sections": sections,
    }


# ===================================================================
# 5) EXECUTIVE BRIEFING (NEW)
# ===================================================================
def build_executive_briefing(  # noqa: C901, PLR0913, PLR0912
    ba: str,
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    anomaly_alerts: pd.DataFrame,
    arbitrage_signals: pd.DataFrame,
    transition_scores: pd.DataFrame,
    lmp_alerts: pd.DataFrame | None = None,
) -> dict:
    """Consolidated briefing for a single BA, pulling from all modules."""
    briefing: dict = {
        "ba": ba,
        "ba_name": BA_METADATA.get(ba, {}).get("name", ba),
        "iso": BA_METADATA.get(ba, {}).get("iso"),
    }

    anomaly_status = "NO_DATA"
    if anomaly_alerts is not None and not anomaly_alerts.empty:
        ba_alert = anomaly_alerts[anomaly_alerts["ba"] == ba]
        if not ba_alert.empty:
            anomaly_status = ba_alert.iloc[0]["status"]
            briefing["latest_error_mwh"] = round(float(ba_alert.iloc[0]["latest_error"]), 0)
    briefing["anomaly_status"] = anomaly_status

    pivot = demand_df[
        (demand_df["respondent"] == ba)
        & (demand_df["type-name"].isin(["Demand", "Day-ahead demand forecast"]))
    ].copy()
    if not pivot.empty:
        pivot = pivot.pivot_table(index="period", columns="type-name", values="value").sort_index(
            ascending=False
        )
        if "Demand" in pivot.columns and len(pivot) > 0:
            briefing["latest_demand_mwh"] = round(float(pivot["Demand"].iloc[0]), 0)
            if "Day-ahead demand forecast" in pivot.columns:
                briefing["latest_forecast_mwh"] = round(
                    float(pivot["Day-ahead demand forecast"].iloc[0]), 0
                )

    if arbitrage_signals is not None and not arbitrage_signals.empty:
        from_ba = arbitrage_signals[arbitrage_signals["fromba"] == ba]
        if not from_ba.empty:
            top = from_ba.iloc[0]
            briefing["top_arbitrage_route"] = f"{top['fromba']} → {top['toba']}"
            briefing["top_arbitrage_score"] = round(float(top["signal_score"]), 1)

    if transition_scores is not None and not transition_scores.empty:
        ba_trans = transition_scores[transition_scores["ba"] == ba]
        if not ba_trans.empty:
            row = ba_trans.iloc[0]
            briefing["transition_score"] = round(float(row["composite_score"]), 1)
            rank_idx = transition_scores.index[transition_scores["ba"] == ba].tolist()
            if rank_idx:
                briefing["transition_rank"] = int(rank_idx[0]) + 1
            briefing["active_queue_mw"] = round(float(row.get("active_capacity_mw", 0) or 0), 0)
            briefing["active_projects"] = int(row.get("active_projects", 0) or 0)

    if not fuel_df.empty and "fueltype" in fuel_df.columns:
        ba_fuel = fuel_df[fuel_df["respondent"] == ba]
        if not ba_fuel.empty:
            total = ba_fuel["value"].sum()
            if total > 0:
                renew = ba_fuel[ba_fuel["fueltype"].isin(RENEWABLE_FUELS)]["value"].sum()
                briefing["renewable_share_pct"] = round((renew / total) * 100, 1)

    iso = BA_METADATA.get(ba, {}).get("iso")
    if iso and lmp_alerts is not None and not lmp_alerts.empty:
        iso_lmps = lmp_alerts[lmp_alerts["iso"] == iso]
        if not iso_lmps.empty:
            briefing["lmp_spike_locations"] = int((iso_lmps["status"] == "SPIKE").sum())
            briefing["lmp_negative_locations"] = int((iso_lmps["status"] == "NEGATIVE").sum())

    return briefing


# ===================================================================
# 6) CORE UTILITIES
# ===================================================================
def prepare_demand_pivot(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
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
    if df_pivot.empty:
        return None, None, None
    try:
        last_actual = df_pivot["Demand"].iloc[0]
        last_forecast = df_pivot["Day-ahead demand forecast"].iloc[0]
        delta = last_actual - last_forecast
    except (KeyError, IndexError):
        return None, None, None
    return last_actual, last_forecast, delta


def compute_generation_mix(df: pd.DataFrame, ba: str | None = None) -> pd.DataFrame:
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


def build_geographic_summary(
    demand_df: pd.DataFrame,
    interchange_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
) -> pd.DataFrame:
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


# Backwards-compat alias
compute_renewable_siting_scores = compute_transition_scores
