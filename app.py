"""Grid Intelligence Platform — Open-source power market intelligence."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import (
    build_executive_briefing,
    compute_interchange_patterns,
    compute_lmp_zonal_spreads,
    compute_transition_scores,
    detect_anomalies,
    detect_lmp_anomalies,
    generate_compliance_summary,
    get_ba_error_distribution,
    get_lmp_time_series,
    get_pair_hourly_profile,
    get_queue_breakdown_for_ba,
    identify_arbitrage_opportunities,
)

# ==========================================
# Config & theme
# ==========================================
st.set_page_config(
    page_title="Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

C_PRIMARY = "#1e40af"
C_SECONDARY = "#0369a1"
C_ACCENT = "#0891b2"
C_RED = "#dc2626"
C_AMBER = "#d97706"
C_GREEN = "#059669"
C_SLATE = "#475569"
C_GRID = "#e2e8f0"
C_MUTED = "#64748b"

FUEL_COLORS = {
    "Natural Gas": "#fb923c",
    "Solar": "#facc15",
    "Wind": "#22d3ee",
    "Nuclear": "#a78bfa",
    "Coal": "#78716c",
    "Hydro": "#38bdf8",
    "Oil": "#f87171",
    "Other": "#cbd5e1",
}
FUEL_LABELS = {
    "NG": "Natural Gas",
    "SUN": "Solar",
    "WND": "Wind",
    "NUC": "Nuclear",
    "COL": "Coal",
    "WAT": "Hydro",
    "OIL": "Oil",
    "OTH": "Other",
}

GCP_PROJECT = "sipa-adv-c-silly-penguin"
BQ_DATASET = "eia_data"
_DS = f"{GCP_PROJECT}.{BQ_DATASET}"

MAJOR_BA = [
    "CISO",
    "ERCO",
    "PJM",
    "MISO",
    "NYIS",
    "ISNE",
    "SWPP",
    "SOCO",
    "TVA",
    "BPAT",
]
BA_NAMES = {
    "CISO": "California ISO",
    "ERCO": "ERCOT (Texas)",
    "PJM": "PJM Interconnection",
    "MISO": "Midcontinent ISO",
    "NYIS": "New York ISO",
    "ISNE": "ISO New England",
    "SWPP": "Southwest Power Pool",
    "SOCO": "Southern Company",
    "TVA": "Tennessee Valley Authority",
    "BPAT": "Bonneville Power Admin",
}
BA_LATLON = {
    "CISO": (36.78, -119.42),
    "ERCO": (31.97, -99.90),
    "PJM": (39.95, -75.16),
    "MISO": (41.88, -87.63),
    "NYIS": (40.71, -74.01),
    "ISNE": (42.36, -71.06),
    "SWPP": (35.47, -97.52),
    "SOCO": (33.75, -84.39),
    "TVA": (36.16, -86.78),
    "BPAT": (45.52, -122.68),
}

NERC_PEAK_START = 14
NERC_PEAK_END = 20

PAGE_BRIEFING = "📊 Executive Briefing"
PAGE_ANOMALY = "🚨 Anomaly Detection"
PAGE_ARBITRAGE = "💰 Arbitrage Signals"
PAGE_TRANSITION = "🌱 Transition Scoring"
PAGE_COMPLIANCE = "📋 Compliance Reports"
PAGE_ABOUT = "📄 About"

PAGES = [
    PAGE_BRIEFING,
    PAGE_ANOMALY,
    PAGE_ARBITRAGE,
    PAGE_TRANSITION,
    PAGE_COMPLIANCE,
    PAGE_ABOUT,
]


# ==========================================
# Plot styling helper
# ==========================================
def _style(fig, h=440):
    """Apply consistent styling to all Plotly figures."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
        margin=dict(l=50, r=20, t=48, b=48),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=C_GRID),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=h,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=C_GRID,
        zeroline=False,
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=C_GRID,
        zeroline=False,
        linecolor="#cbd5e1",
    )
    return fig


def _cross_ref(text: str) -> None:
    """Render a cross-module navigation hint at the top of a module."""
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:0.88rem;"
        f"padding:0.35rem 0 0.6rem 0;'>↪ {text}</div>",
        unsafe_allow_html=True,
    )


def _source(text: str) -> None:
    """Render a source attribution line below a chart.

    Appears as italic grey text. Use for data provenance only; keep
    methodology/formulas in `_methodology()` instead.
    """
    st.markdown(
        f"<div style='color:{C_MUTED};font-size:0.78rem;"
        f"font-style:italic;margin:-0.3rem 0 0.2rem 0;'>Source: {text}</div>",
        unsafe_allow_html=True,
    )


def _methodology(text: str) -> None:
    """Render a methodology / formula block below a chart.

    Renders as a bordered light-grey panel with monospace-friendly layout
    for formulas. Accepts inline HTML (useful for <code> tags and <br/>).
    """
    st.markdown(
        f"<details style='margin:0.3rem 0 0.8rem 0;'>"
        f"<summary style='color:{C_PRIMARY};font-size:0.82rem;"
        f"font-weight:600;cursor:pointer;'>▸ Methodology & formula</summary>"
        f"<div style='background:#f8fafc;border-left:3px solid {C_PRIMARY};"
        f"padding:0.7rem 1rem;margin-top:0.4rem;font-size:0.85rem;"
        f"color:#334155;line-height:1.6;'>{text}</div>"
        f"</details>",
        unsafe_allow_html=True,
    )


def _section_divider() -> None:
    st.markdown(
        f"<hr style='border:none;border-top:1px solid {C_GRID};margin:1.2rem 0;'/>",
        unsafe_allow_html=True,
    )


def _status_pill(status: str) -> str:
    """Return a colored pill for a status string."""
    color_map = {
        "RED": C_RED,
        "YELLOW": C_AMBER,
        "NORMAL": C_GREEN,
        "GREEN": C_GREEN,
        "NO_DATA": C_MUTED,
        "SPIKE": C_RED,
        "NEGATIVE": C_AMBER,
    }
    label_map = {
        "RED": "CRITICAL",
        "YELLOW": "WARNING",
        "NORMAL": "NORMAL",
        "GREEN": "NORMAL",
        "NO_DATA": "NO DATA",
        "SPIKE": "SPIKE",
        "NEGATIVE": "NEGATIVE",
    }
    color = color_map.get(status, C_MUTED)
    label = label_map.get(status, status)
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.78rem;font-weight:600;'>{label}</span>"
    )


# ==========================================
# Auth & data loading
# ==========================================
def _get_creds():
    try:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
    except (FileNotFoundError, KeyError):
        return None


_creds = _get_creds()


@st.cache_resource(ttl=3600, show_spinner="⚡ Loading market data...")
def _load_all() -> dict:
    """One-shot load of all BigQuery tables into memory."""
    from google.cloud import bigquery

    cl = bigquery.Client(project=GCP_PROJECT, credentials=_creds)

    def q(s: str) -> pd.DataFrame:
        return cl.query(s).to_dataframe()

    def q_safe(s: str) -> pd.DataFrame:
        """Tolerant query — returns empty DataFrame if the table is missing."""
        try:
            return cl.query(s).to_dataframe()
        except Exception:
            return pd.DataFrame()

    d: dict = {}

    df = q(
        "SELECT period, respondent, `type-name`, "
        "CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_demand`"
    )
    df["period"] = pd.to_datetime(df["period"])
    d["demand"] = df

    df = q(
        "SELECT period, fromba, toba, "
        "CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_interchange`"
    )
    df["period"] = pd.to_datetime(df["period"])
    d["interchange"] = df

    df = q(
        "SELECT period, respondent, fueltype, "
        "CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_fuel_type`"
    )
    df["period"] = pd.to_datetime(df["period"])
    d["fuel"] = df

    df = q(
        f"SELECT date, CAST(ng_price AS FLOAT64) AS ng_price FROM `{_DS}.daily_ng_price`"
    )
    df["date"] = pd.to_datetime(df["date"])
    d["ng_price"] = df

    d["weather"] = q_safe(
        "SELECT date, ba, "
        "CAST(avg_temp AS FLOAT64) AS avg_temp, "
        "CAST(max_temp AS FLOAT64) AS max_temp, "
        "CAST(min_temp AS FLOAT64) AS min_temp "
        f"FROM `{_DS}.daily_weather`"
    )
    if not d["weather"].empty:
        d["weather"]["date"] = pd.to_datetime(d["weather"]["date"])

    d["ranking"] = q_safe(f"SELECT * FROM `{_DS}.ba_mape_ranking` ORDER BY mape")

    d["queue_ba"] = q_safe(f"SELECT * FROM `{_DS}.queue_ba_summary`")
    d["queue_type"] = q_safe(f"SELECT * FROM `{_DS}.queue_type_summary`")
    d["nrel"] = q_safe(f"SELECT * FROM `{_DS}.nrel_resource_locations`")

    d["lmp"] = q_safe(
        "SELECT iso, time, location, location_type, "
        "CAST(lmp AS FLOAT64) AS lmp, "
        "CAST(congestion AS FLOAT64) AS congestion "
        f"FROM `{_DS}.iso_hourly_lmp`"
    )
    if not d["lmp"].empty:
        d["lmp"]["time"] = pd.to_datetime(d["lmp"]["time"])

    return d


_D = _load_all()


# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("⚡ Grid Intelligence")
st.sidebar.caption("Power market intelligence — built on public data")
st.sidebar.markdown("---")
page = st.sidebar.radio("Module", PAGES)
st.sidebar.markdown("---")
sel_ba = st.sidebar.selectbox(
    "Balancing Authority",
    MAJOR_BA,
    format_func=lambda x: f"{x} — {BA_NAMES[x]}",
)
days = st.sidebar.slider("Time window (days)", 1, 30, 7)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data freshness**: ETL refreshes daily at 09:00 UTC via GitHub Actions. "
    "Dashboard cache TTL = 1h."
)


# ==========================================
# Pre-compute cross-module signals (cheap, reused across pages)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def _compute_signals() -> dict:
    alerts = detect_anomalies(_D["demand"])
    arb = identify_arbitrage_opportunities(_D["interchange"])
    trans = compute_transition_scores(
        _D["demand"],
        _D["fuel"],
        _D["interchange"],
        queue_summary=_D["queue_ba"] if not _D["queue_ba"].empty else None,
    )
    lmp_alerts = (
        detect_lmp_anomalies(_D["lmp"]) if not _D["lmp"].empty else pd.DataFrame()
    )
    lmp_spreads = (
        compute_lmp_zonal_spreads(_D["lmp"]) if not _D["lmp"].empty else pd.DataFrame()
    )
    return {
        "alerts": alerts,
        "arbitrage": arb,
        "transition": trans,
        "lmp_alerts": lmp_alerts,
        "lmp_spreads": lmp_spreads,
    }


_S = _compute_signals()


# ===================================================================
# 📊 EXECUTIVE BRIEFING
# ===================================================================
if page == PAGE_BRIEFING:
    t0 = time.time()
    st.title("📊 Executive Briefing")
    _cross_ref(
        f"Single-pane snapshot of <b>{sel_ba}</b>. Drill into modules below for full analysis."
    )

    briefing = build_executive_briefing(
        ba=sel_ba,
        demand_df=_D["demand"],
        interchange_df=_D["interchange"],
        fuel_df=_D["fuel"],
        anomaly_alerts=_S["alerts"],
        arbitrage_signals=_S["arbitrage"],
        transition_scores=_S["transition"],
        lmp_alerts=_S["lmp_alerts"],
    )

    st.markdown(
        f"## {briefing['ba']} — {briefing['ba_name']}",
        help="Selected balancing authority. Change in sidebar.",
    )

    # Headline status row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("**Forecast Status**")
        st.markdown(
            _status_pill(briefing.get("anomaly_status", "NO_DATA")),
            unsafe_allow_html=True,
        )
        if briefing.get("latest_error_mwh") is not None:
            st.caption(f"Latest error: {briefing['latest_error_mwh']:,.0f} MWh")
    with k2:
        latest = briefing.get("latest_demand_mwh")
        forecast = briefing.get("latest_forecast_mwh")
        if latest is not None:
            delta = (latest - forecast) if forecast is not None else None
            st.metric(
                "Latest Demand",
                f"{latest:,.0f} MWh",
                delta=f"{delta:+,.0f} vs forecast" if delta is not None else None,
                delta_color="inverse",
            )
        else:
            st.metric("Latest Demand", "—")
    with k3:
        score = briefing.get("transition_score")
        rank = briefing.get("transition_rank")
        if score is not None:
            st.metric(
                "Transition Score",
                f"{score:.1f}",
                delta=f"Rank {rank} of {len(MAJOR_BA)}" if rank else None,
                delta_color="off",
            )
        else:
            st.metric("Transition Score", "—")
    with k4:
        renew = briefing.get("renewable_share_pct")
        if renew is not None:
            st.metric("Renewable Share", f"{renew:.1f}%")
        else:
            st.metric("Renewable Share", "—")

    # LMP context line — only renders when this BA maps to a covered ISO
    spikes = briefing.get("lmp_spike_locations")
    negs = briefing.get("lmp_negative_locations")
    if spikes is not None or negs is not None:
        iso_label = briefing.get("iso", "")
        bits = []
        if spikes:
            bits.append(
                f"<span style='color:{C_RED};font-weight:600;'>{spikes} spike</span>"
            )
        if negs:
            bits.append(
                f"<span style='color:{C_AMBER};font-weight:600;'>{negs} negative</span>"
            )
        if not bits:
            bits.append(f"<span style='color:{C_GREEN};'>no LMP anomalies</span>")
        st.markdown(
            f"<div style='color:{C_MUTED};font-size:0.88rem;margin-top:0.5rem;'>"
            f"⚡ <b>{iso_label} LMP:</b> {' · '.join(bits)} in last 6h "
            f"(see Anomaly Detection)</div>",
            unsafe_allow_html=True,
        )

    _section_divider()

    # Module preview row 1: forecast + arbitrage
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(
            "Forecast Error — Recent",
            help=(
                "Absolute day-ahead forecast error for the selected BA over the "
                "sidebar-configurable time window. The red dashed line shows the "
                "BA's P95 control limit — bars touching or crossing it are the "
                "worst 5% of errors in the BA's own distribution. Sustained "
                "breaches (≥3 hours out of the last 6) trigger the RED status "
                "in the Anomaly Detection module."
            ),
        )
        ba_err = get_ba_error_distribution(_D["demand"], sel_ba)
        if ba_err.empty:
            st.info("No forecast error data for this BA.")
        else:
            recent = ba_err.tail(days * 24)
            ba_alert = _S["alerts"][_S["alerts"]["ba"] == sel_ba]
            p95 = ba_alert["p95_threshold"].iloc[0] if not ba_alert.empty else None
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=recent["period"],
                    y=recent["abs_error"],
                    mode="lines",
                    line=dict(color=C_PRIMARY, width=1.4),
                    fill="tozeroy",
                    fillcolor="rgba(30,64,175,0.06)",
                    name="Abs Error",
                )
            )
            if p95:
                fig.add_hline(
                    y=p95,
                    line_dash="dash",
                    line_color=C_RED,
                    line_width=1.2,
                    annotation_text=f"<b>P95 = {p95:,.0f}</b>",
                    annotation_font=dict(color=C_RED, size=10),
                    annotation_bgcolor="white",
                    annotation_bordercolor=C_RED,
                    annotation_borderwidth=1,
                    annotation_borderpad=3,
                    annotation_position="top right",
                )
            fig.update_layout(xaxis_title="", yaxis_title="MWh")
            _style(fig, h=300)
            st.plotly_chart(fig, use_container_width=True)
            _source(f"EIA Form 930 hourly demand and day-ahead forecast for {sel_ba}.")
            _methodology(
                "Plotted series: <code>abs_error<sub>t</sub> = |demand<sub>t</sub> − "
                "forecast<sub>t</sub>|</code> for the last "
                f"{days} days. The P95 reference line is the 95th percentile "
                "of <code>abs_error</code> computed over the full 3-month "
                "history for this BA."
            )
        st.caption(
            "→ Full control chart and cross-BA comparison in **Anomaly Detection**"
        )

    with col_b:
        st.subheader(
            "Top Arbitrage Routes",
            help=(
                "Ranks outbound interchange routes from this BA by a composite "
                "arbitrage signal score during NERC peak hours (14:00–20:00). "
                "High-scoring routes combine large directional flow magnitude "
                "with low hour-to-hour volatility, indicating a structural "
                "price-arbitrage opportunity rather than sporadic balancing flow."
            ),
        )
        arb = _S["arbitrage"]
        ba_arb = (
            arb[arb["fromba"] == sel_ba].head(5) if not arb.empty else pd.DataFrame()
        )
        if ba_arb.empty:
            st.info("No outgoing arbitrage signals for this BA.")
        else:
            ba_arb = ba_arb.copy()
            ba_arb["route"] = ba_arb["fromba"] + " → " + ba_arb["toba"]
            fig = go.Figure(
                go.Bar(
                    x=ba_arb["signal_score"],
                    y=ba_arb["route"],
                    orientation="h",
                    marker_color=C_SECONDARY,
                    text=ba_arb["signal_score"].round(1),
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Signal Score (0–100)",
                yaxis_title="",
                yaxis=dict(categoryorder="total ascending"),
            )
            _style(fig, h=300)
            st.plotly_chart(fig, use_container_width=True)
            _source(
                "EIA Form 930 hourly interchange data for all BA pairs where "
                f"<code>fromba = {sel_ba}</code>."
            )
            _methodology(
                "See full derivation in the Arbitrage Signals module. "
                "Score combines <b>directional strength</b> (60%, normalized "
                "peak-hour absolute flow) and <b>consistency</b> (40%, "
                "<code>1 − volatility/max_volatility</code>), scaled to 0–100."
            )
        st.caption("→ Heatmap and route profiles in **Arbitrage Signals**")

    _section_divider()

    # Module preview row 2: transition + queue
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader(
            "Transition Score Profile",
            help=(
                "Radar plot of the six factors that compose this BA's renewable "
                "investment opportunity score. Each axis runs 0–100, where higher "
                "values mean stronger opportunity on that dimension. The shaded "
                "polygon area is a rough visual proxy for total opportunity; "
                "asymmetric shapes indicate a BA that scores well on some "
                "factors but poorly on others (e.g., strong demand growth but "
                "weak queue pipeline)."
            ),
        )
        ba_t = _S["transition"][_S["transition"]["ba"] == sel_ba]
        if ba_t.empty:
            st.info("Transition scores not available.")
        else:
            cats = [
                "Demand Growth",
                "Renewable Headroom",
                "Import Dependence",
                "Fossil Transition",
                "Queue Activity",
                "Queue Completion",
            ]
            row = ba_t.iloc[0]
            vals = [
                float(row["demand_growth_score"]),
                float(row["renewable_headroom_score"]),
                float(row["import_dependence_score"]),
                float(row["fossil_transition_score"]),
                float(row["queue_active_score"])
                if pd.notna(row["queue_active_score"])
                else 0.0,
                (
                    float(row["queue_completion_score"])
                    if pd.notna(row["queue_completion_score"])
                    else 0.0
                ),
            ]
            fig = go.Figure(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(5,150,105,0.12)",
                    line=dict(color=C_GREEN, width=2),
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    bgcolor="white",
                ),
                showlegend=False,
            )
            _style(fig, h=320)
            st.plotly_chart(fig, use_container_width=True)
            _source(
                "EIA Form 930 (demand, fuel mix, interchange) + LBNL "
                "interconnection queue (Berkeley Lab, thru 2024). Full derivation "
                "of each axis is documented in the Transition Scoring module."
            )
        st.caption("→ Composite ranking and queue detail in **Transition Scoring**")

    with col_d:
        st.subheader(
            "Interconnection Queue",
            help=(
                "Summary of the generation and storage projects in the "
                "interconnection queue for this BA, based on LBNL's annual "
                "dataset. 'Active' means projects currently under study or "
                "awaiting IA execution. The breakdown by resource type shows "
                "where developer demand is concentrated — heavy solar/storage "
                "bias is typical of high-transition-score BAs."
            ),
        )
        if briefing.get("active_projects"):
            qm1, qm2 = st.columns(2)
            qm1.metric("Active Projects", f"{briefing['active_projects']:,}")
            qm2.metric(
                "Active Capacity", f"{briefing.get('active_queue_mw', 0):,.0f} MW"
            )
        if not _D["queue_type"].empty:
            qb = get_queue_breakdown_for_ba(_D["queue_type"], sel_ba)
            if not qb.empty:
                qb = qb.head(8).copy()
                fig = go.Figure(
                    go.Bar(
                        x=qb["active_mw"],
                        y=qb["resource_type"],
                        orientation="h",
                        marker_color=C_GREEN,
                        text=qb["active_mw"].round(0).astype(int),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    xaxis_title="Active MW",
                    yaxis_title="",
                    yaxis=dict(categoryorder="total ascending"),
                )
                _style(fig, h=240)
                st.plotly_chart(fig, use_container_width=True)
                _source(
                    "LBNL Queued Up report "
                    "(<code>interconnection_queue</code> table, thru 2024), "
                    f"filtered to projects mapped to BA {sel_ba} by "
                    "ISO/region affiliation or state."
                )
            else:
                st.caption("No queue breakdown available for this BA.")
        st.caption("→ Resource-type breakdown details in **Transition Scoring**")

    _section_divider()

    # Compliance teaser
    st.subheader(
        "Compliance Snapshot",
        help=(
            "Four high-level regulatory KPIs extracted from this BA's compliance "
            "report: average and peak demand (operational scale), MAPE (forecast "
            "accuracy — see below), and number of trading partners (interchange "
            "connectivity). The full FERC Form 714 / EIA Form 930-style report "
            "with all five sections (including cross-module signals) is available "
            "in the Compliance Reports module."
        ),
    )
    report = generate_compliance_summary(
        _D["demand"],
        _D["interchange"],
        _D["fuel"],
        sel_ba,
        anomaly_alerts=_S["alerts"],
        transition_scores=_S["transition"],
    )
    sec = report["sections"]
    cc1, cc2, cc3, cc4 = st.columns(4)
    if "demand" in sec:
        cc1.metric("Avg Demand", f"{sec['demand']['avg_demand_mwh']:,.0f} MWh")
        cc2.metric("Peak Demand", f"{sec['demand']['peak_demand_mwh']:,.0f} MWh")
    if "forecast_accuracy" in sec:
        cc3.metric("MAPE", f"{sec['forecast_accuracy']['mape']:.2f}%")
    if "interchange" in sec:
        cc4.metric("Trading Partners", f"{sec['interchange']['n_trading_partners']}")
    _source(
        "EIA Form 930 (hourly demand, forecast, and interchange) aggregated over "
        "the full 3-month window for this BA."
    )
    _methodology(
        "<b>MAPE</b> (Mean Absolute Percentage Error): "
        "<code>MAPE = mean( |demand<sub>t</sub> − forecast<sub>t</sub>| / "
        "demand<sub>t</sub> )</code> × 100. "
        "Hours with <code>demand<sub>t</sub> = 0</code> are excluded to avoid "
        "division by zero. Lower is better. A well-run BA on a stable grid "
        "typically sits in the 1–3% range; values above 5% signal either "
        "forecasting-model issues or unusual weather/load patterns."
    )
    st.caption("→ Full FERC-style sections in **Compliance Reports**")

    st.caption(f"Briefing assembled in {time.time() - t0:.2f}s")


# ===================================================================
# 🚨 ANOMALY DETECTION
# ===================================================================
elif page == PAGE_ANOMALY:
    t0 = time.time()
    st.title("🚨 Anomaly Detection")
    _cross_ref(
        "Flagged BAs feed into <b>Compliance Reports §5</b> and the "
        "<b>Executive Briefing</b> headline status."
    )
    st.markdown(
        "Monitors day-ahead forecast errors across all balancing authorities and "
        "flags regions where recent errors persistently exceed historical norms.",
        help=(
            "For each BA we compute the P90 and P95 of historical absolute forecast "
            "error. If ≥3 of the last 6 hours exceed P95 → RED. ≥3 exceed P90 → YELLOW. "
            "Otherwise NORMAL."
        ),
    )

    alerts = _S["alerts"]
    if alerts.empty:
        st.info("No forecast error data available.")
    else:
        red = alerts[alerts["status"] == "RED"]
        yel = alerts[alerts["status"] == "YELLOW"]
        grn = alerts[alerts["status"] == "NORMAL"]

        col_r, col_y, col_g = st.columns(3)
        with col_r:
            st.markdown(f"#### 🔴 Critical ({len(red)})")
            for _, r in red.iterrows():
                st.error(
                    f"**{r['ba']}** — {BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: **{r['latest_error']:,.0f}** MWh"
                    f" · 6h avg: {r['recent_mean_error']:,.0f}"
                    f" · P95: {r['p95_threshold']:,.0f}",
                    icon="🔴",
                )
            if red.empty:
                st.caption("None.")
        with col_y:
            st.markdown(f"#### 🟡 Warning ({len(yel)})")
            for _, r in yel.iterrows():
                st.warning(
                    f"**{r['ba']}** — {BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: **{r['latest_error']:,.0f}** MWh"
                    f" · 6h avg: {r['recent_mean_error']:,.0f}"
                    f" · P90: {r['p90_threshold']:,.0f}",
                    icon="🟡",
                )
            if yel.empty:
                st.caption("None.")
        with col_g:
            st.markdown(f"#### 🟢 Normal ({len(grn)})")
            for _, r in grn.iterrows():
                st.success(
                    f"**{r['ba']}** — {BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: {r['latest_error']:,.0f} MWh · Within normal range",
                    icon="🟢",
                )

        _section_divider()

        # Control chart
        st.subheader(
            f"{sel_ba} — Error Control Chart",
            help=(
                "Shewhart-style individuals control chart for day-ahead demand "
                "forecast performance. Plots the absolute error |Demand − Forecast| "
                "for every hour in the rolling 3-month window against two control "
                "limits derived from this BA's own historical distribution: P90 "
                "(early-warning) and P95 (out-of-control). Unlike a fixed-MW "
                "threshold, percentile-based limits adapt to each BA's natural "
                "error scale, so a small BA with 500 MWh errors can be flagged "
                "while PJM is not — even though PJM's absolute errors are larger."
            ),
        )
        ba_err = get_ba_error_distribution(_D["demand"], sel_ba)
        if ba_err.empty:
            st.info(f"No forecast data for {sel_ba}.")
        else:
            ba_a = alerts[alerts["ba"] == sel_ba]
            p95 = ba_a["p95_threshold"].iloc[0] if not ba_a.empty else None
            p90 = ba_a["p90_threshold"].iloc[0] if not ba_a.empty else None
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ba_err["period"],
                    y=ba_err["abs_error"],
                    mode="lines",
                    name="Absolute Error",
                    line=dict(color=C_PRIMARY, width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(30,64,175,0.06)",
                )
            )
            if p95:
                fig.add_hline(
                    y=p95,
                    line_dash="dash",
                    line_color=C_RED,
                    line_width=1.5,
                    annotation_text=f"<b>P95 = {p95:,.0f}</b>",
                    annotation_font=dict(color=C_RED, size=11),
                    annotation_bgcolor="white",
                    annotation_bordercolor=C_RED,
                    annotation_borderwidth=1,
                    annotation_borderpad=4,
                    annotation_position="top right",
                )
            if p90:
                fig.add_hline(
                    y=p90,
                    line_dash="dot",
                    line_color=C_AMBER,
                    line_width=1.2,
                    annotation_text=f"<b>P90 = {p90:,.0f}</b>",
                    annotation_font=dict(color=C_AMBER, size=11),
                    annotation_bgcolor="white",
                    annotation_bordercolor=C_AMBER,
                    annotation_borderwidth=1,
                    annotation_borderpad=4,
                    annotation_position="bottom right",
                )
            fig.update_layout(xaxis_title="", yaxis_title="Absolute Error (MWh)")
            _style(fig)
            st.plotly_chart(fig, use_container_width=True)
            _source(
                f"EIA Form 930 hourly demand and day-ahead forecast for {sel_ba}, "
                "retrieved via EIA API v2 (<code>electricity/rto/region-data</code>)."
            )
            _methodology(
                "<b>Absolute forecast error</b>: "
                "<code>abs_error<sub>t</sub> = |demand<sub>t</sub> − "
                "forecast<sub>t</sub>|</code><br/>"
                "<b>Control limits</b>: P90 and P95 are the 90th and 95th "
                "percentiles of the full historical <code>abs_error</code> "
                "series for this BA over the rolling 3-month window.<br/>"
                "<b>Alert rule</b> (Shewhart-style, applied to the last 6 hours): "
                "<code>RED</code> if ≥3 hours exceed P95; <code>YELLOW</code> if "
                "≥3 exceed P90; otherwise <code>NORMAL</code>. The rule requires "
                "<i>sustained</i> deviation, not a single outlier, to avoid false "
                "alarms from isolated data-entry errors."
            )

        # Violin
        st.subheader(
            "Cross-BA Error Comparison",
            help=(
                "Kernel-density violin plots comparing the full distribution of "
                "absolute forecast errors across all 10 tracked BAs. The embedded "
                "box shows median and IQR; the violin width at each y-value is "
                "proportional to the density of hours at that error level. "
                "Y-axis is logarithmic because error magnitudes span two to three "
                "orders of magnitude across BAs (small BAs like BPAT vs. large "
                "ones like PJM), and linear scale would compress the smaller BAs "
                "into an unreadable strip."
            ),
        )
        all_err = []
        for ba in MAJOR_BA:
            e = get_ba_error_distribution(_D["demand"], ba)
            if not e.empty:
                e = e[["abs_error"]].copy()
                e["BA"] = ba
                all_err.append(e)
        if all_err:
            err_df = pd.concat(all_err, ignore_index=True)
            # Drop non-positive values so log scale doesn't explode
            err_df = err_df[err_df["abs_error"] > 0]
            # Order BAs by median error so the plot reads top-to-bottom best→worst
            median_order = (
                err_df.groupby("BA")["abs_error"].median().sort_values().index.tolist()
            )
            fig_v = px.violin(
                err_df,
                y="BA",
                x="abs_error",
                box=True,
                points=False,
                orientation="h",
                color_discrete_sequence=[C_PRIMARY],
                labels={"abs_error": "Absolute Error (MWh, log scale)", "BA": ""},
                category_orders={"BA": median_order},
                log_x=True,
            )
            fig_v.update_traces(width=0.85, meanline_visible=True)
            _style(fig_v, h=560)
            st.plotly_chart(fig_v, use_container_width=True)
            _source(
                "EIA Form 930 hourly demand and day-ahead forecast across all 10 "
                "tracked BAs, retrieved via EIA API v2."
            )
            _methodology(
                "<b>Distribution</b>: for each BA, all hourly "
                "<code>abs_error<sub>t</sub> = |demand<sub>t</sub> − "
                "forecast<sub>t</sub>|</code> values in the rolling 3-month "
                "window.<br/>"
                "<b>Violin width</b>: Gaussian kernel density estimate of the "
                "error distribution. Wider regions indicate error magnitudes "
                "that occur more frequently.<br/>"
                "<b>Box elements</b>: box edges = Q1 and Q3; center line = "
                "median; whiskers = 1.5 × IQR beyond the box.<br/>"
                "<b>Ordering</b>: BAs are sorted top-to-bottom by median "
                "absolute error, lowest first. Reading down the list is equivalent "
                "to scanning from best- to worst-performing forecasts in absolute "
                "terms (before normalizing by demand size)."
            )

        _section_divider()

        # ---------- LMP price anomalies ----------
        st.subheader(
            "LMP Price Anomalies",
            help=(
                "Detects price dislocations at ISO pricing nodes (zones and hubs). "
                "Two event classes are flagged: <b>SPIKE</b> — sustained high prices, "
                "typically driven by scarcity, transmission congestion, or supply "
                "shortfalls; and <b>NEGATIVE</b> — sub-zero prices, which occur "
                "when must-run or subsidized generation (wind, nuclear, thermal "
                "minimums) exceeds load and generators pay the grid to keep "
                "producing. Both signal real dispatch and hedging opportunities "
                "for market participants with flexible load or storage."
            ),
        )

        lmp_alerts = _S["lmp_alerts"]
        if lmp_alerts.empty:
            st.info(
                "ISO LMP table is empty. Run the ETL — `iso_hourly_lmp` is populated "
                "from gridstatus (CAISO + ERCOT). PJM coverage requires "
                "`PJM_API_KEY` in repo secrets."
            )
        else:
            spike = lmp_alerts[lmp_alerts["status"] == "SPIKE"]
            neg = lmp_alerts[lmp_alerts["status"] == "NEGATIVE"]
            norm = lmp_alerts[lmp_alerts["status"] == "NORMAL"]

            l_r, l_y, l_g = st.columns(3)
            with l_r:
                st.markdown(f"#### 🔴 Price Spike ({len(spike)})")
                for _, r in spike.head(8).iterrows():
                    st.error(
                        f"**{r['iso']} — {r['location']}**  \n"
                        f"6h avg: **${r['recent_avg_lmp']:,.1f}**/MWh"
                        f" · median: ${r['historical_median']:,.1f}"
                        f" · ratio: {r['spike_ratio']:.1f}×",
                        icon="🔴",
                    )
                if spike.empty:
                    st.caption("None.")
            with l_y:
                st.markdown(f"#### 🟡 Negative Prices ({len(neg)})")
                for _, r in neg.head(8).iterrows():
                    st.warning(
                        f"**{r['iso']} — {r['location']}**  \n"
                        f"Min: **${r['recent_min_lmp']:,.1f}**/MWh"
                        f" · 6h avg: ${r['recent_avg_lmp']:,.1f}",
                        icon="🟡",
                    )
                if neg.empty:
                    st.caption("None.")
            with l_g:
                st.markdown(f"#### 🟢 Normal ({len(norm)})")
                for _, r in norm.head(8).iterrows():
                    st.success(
                        f"**{r['iso']} — {r['location']}**  \n"
                        f"Latest: ${r['latest_lmp']:,.1f}/MWh",
                        icon="🟢",
                    )
                if norm.empty:
                    st.caption("None.")

            _source(
                "Day-ahead hourly LMP for CAISO and ERCOT zones/hubs, retrieved "
                "via the <code>gridstatus</code> Python library "
                "(CAISO OASIS <code>PRC_LMP</code> API and ERCOT MIS reports)."
            )
            _methodology(
                "<b>Inputs</b>: hourly day-ahead LMP series "
                "(<code>lmp<sub>t</sub></code>) for each ISO × location pair over "
                "the last 30 days. Only <code>ZONE</code>, <code>HUB</code>, and "
                "<code>TRADING_HUB</code> locations are retained; resource-node "
                "LMPs are excluded.<br/>"
                "<b>Baseline</b>: <code>median<sub>hist</sub></code> = median of "
                "all historical LMP observations for the location. Locations with "
                "non-positive medians are dropped to avoid undefined spike "
                "ratios.<br/>"
                "<b>Recent window</b>: last 6 hours of LMP data. "
                "<code>avg<sub>6h</sub></code>, <code>min<sub>6h</sub></code>, "
                "and <code>max<sub>6h</sub></code> are the mean, min, and max "
                "over this window.<br/>"
                "<b>Classification</b>:<br/>"
                "&nbsp;&nbsp;• <code>SPIKE</code> if "
                "<code>avg<sub>6h</sub> > 3 × median<sub>hist</sub></code>"
                " (multiplier = <code>LMP_SPIKE_MULTIPLIER</code>)<br/>"
                "&nbsp;&nbsp;• <code>NEGATIVE</code> if "
                "<code>min<sub>6h</sub> < −$10/MWh</code>"
                " (threshold = <code>LMP_NEGATIVE_THRESHOLD</code>)<br/>"
                "&nbsp;&nbsp;• <code>NORMAL</code> otherwise. SPIKE takes priority "
                "over NEGATIVE when both conditions apply."
            )

            # Time series for a selected location
            st.markdown("**LMP time series — drill down**")
            iso_options = sorted(lmp_alerts["iso"].unique().tolist())
            iso_pick = st.selectbox("ISO", iso_options, key="lmp_iso")
            loc_options = sorted(
                lmp_alerts[lmp_alerts["iso"] == iso_pick]["location"].unique().tolist()
            )
            loc_pick = st.selectbox("Location", loc_options, key="lmp_loc")
            ts = get_lmp_time_series(_D["lmp"], iso_pick, loc_pick)
            if not ts.empty:
                row = lmp_alerts[
                    (lmp_alerts["iso"] == iso_pick)
                    & (lmp_alerts["location"] == loc_pick)
                ]
                median = (
                    float(row["historical_median"].iloc[0]) if not row.empty else None
                )
                fig_lmp = go.Figure()
                fig_lmp.add_trace(
                    go.Scatter(
                        x=ts["time"],
                        y=ts["lmp"],
                        mode="lines",
                        line=dict(color=C_PRIMARY, width=1.2),
                        name="LMP",
                    )
                )
                if median is not None:
                    fig_lmp.add_hline(
                        y=median,
                        line_dash="dot",
                        line_color=C_SLATE,
                        annotation_text=f"<b>Median = ${median:,.1f}</b>",
                        annotation_font=dict(color=C_SLATE, size=10),
                        annotation_bgcolor="white",
                        annotation_bordercolor=C_SLATE,
                        annotation_borderwidth=1,
                        annotation_borderpad=4,
                        annotation_position="top left",
                    )
                    fig_lmp.add_hline(
                        y=median * 3,
                        line_dash="dash",
                        line_color=C_RED,
                        annotation_text=f"<b>Spike threshold = ${median * 3:,.1f}</b>",
                        annotation_font=dict(color=C_RED, size=10),
                        annotation_bgcolor="white",
                        annotation_bordercolor=C_RED,
                        annotation_borderwidth=1,
                        annotation_borderpad=4,
                        annotation_position="top right",
                    )
                fig_lmp.add_hline(
                    y=-10,
                    line_dash="dash",
                    line_color=C_AMBER,
                    annotation_text="<b>Negative threshold = −$10</b>",
                    annotation_font=dict(color=C_AMBER, size=10),
                    annotation_bgcolor="white",
                    annotation_bordercolor=C_AMBER,
                    annotation_borderwidth=1,
                    annotation_borderpad=4,
                    annotation_position="bottom right",
                )
                fig_lmp.update_layout(xaxis_title="", yaxis_title="LMP ($/MWh)")
                _style(fig_lmp, h=380)
                st.plotly_chart(fig_lmp, use_container_width=True)
                _source(
                    f"Day-ahead hourly LMP for {iso_pick} {loc_pick}, retrieved "
                    "via the <code>gridstatus</code> library over a 30-day window."
                )
                _methodology(
                    "The three horizontal reference lines show the thresholds "
                    "applied by the classifier above:<br/>"
                    "&nbsp;&nbsp;• <b>Median line</b> = historical median LMP for "
                    "this location across the full 30-day window.<br/>"
                    "&nbsp;&nbsp;• <b>Spike threshold</b> = <code>3 × median</code>. "
                    "Hours with 6-hour rolling average above this line feed the "
                    "SPIKE classification.<br/>"
                    "&nbsp;&nbsp;• <b>Negative threshold</b> = <code>−$10/MWh</code>. "
                    "Any recent hour below this line feeds the NEGATIVE "
                    "classification."
                )

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 💰 ARBITRAGE SIGNALS
# ===================================================================
elif page == PAGE_ARBITRAGE:
    t0 = time.time()
    st.title("💰 Arbitrage Signals")
    _cross_ref(
        "Top route for the selected BA appears in the <b>Executive Briefing</b>. "
        "Inter-BA flow signals are paired with intra-ISO zonal LMP spreads below."
    )
    st.markdown(
        "Identifies persistent directional power flows between regions during peak "
        "hours, indicating potential price differentials.",
        help=(
            "We compute average hourly interchange for every BA pair, focusing on "
            "NERC peak hours (14:00–20:00). Signal Score = 60% directional strength "
            "+ 40% consistency."
        ),
    )

    signals = _S["arbitrage"]
    if signals.empty:
        st.info("No interchange data available.")
    else:
        disp = signals.head(12).copy()
        disp["route"] = disp["fromba"] + " → " + disp["toba"]

        st.subheader(
            "Top Opportunities — NERC Peak (14–20h)",
            help=(
                "Three-panel comparison of the top 8 inter-BA interchange routes "
                "ranked by composite Signal Score. Each panel shows one input to "
                "the score so you can see <i>why</i> a route ranks where it does: "
                "absolute flow magnitude (how much power moves), and flow "
                "consistency (how stable the direction is hour-to-hour). The "
                "NERC peak window (14:00–20:00 local) is when price spreads "
                "are structurally largest, so we isolate flows during that "
                "window rather than averaging across all 24 hours."
            ),
        )
        top8 = disp.head(8).sort_values("signal_score")
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=True,
            subplot_titles=("Signal Score", "Peak Avg Flow (MWh)", "Consistency"),
            horizontal_spacing=0.06,
        )
        fig.add_trace(
            go.Bar(
                x=top8["signal_score"],
                y=top8["route"],
                orientation="h",
                marker_color=C_SECONDARY,
                showlegend=False,
                text=top8["signal_score"].round(1),
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=top8["peak_avg_flow"].abs(),
                y=top8["route"],
                orientation="h",
                marker_color=np.where(top8["peak_avg_flow"] > 0, C_GREEN, C_RED),
                showlegend=False,
                text=top8["peak_avg_flow"].round(0).astype(int),
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        cons = top8.get("consistency", pd.Series([0.5] * len(top8)))
        fig.add_trace(
            go.Bar(
                x=cons,
                y=top8["route"],
                orientation="h",
                marker_color=C_ACCENT,
                showlegend=False,
                text=cons.round(2),
                textposition="outside",
            ),
            row=1,
            col=3,
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        _style(fig, h=420)
        st.plotly_chart(fig, use_container_width=True)
        _source(
            "EIA Form 930 hourly interchange data across all BA pairs, "
            "filtered to NERC peak hours (14:00–20:00)."
        )
        _methodology(
            "<b>Inputs</b> (per BA pair, averaged over peak hours only):<br/>"
            "&nbsp;&nbsp;• <code>peak_flow</code> = mean hourly interchange "
            "in MWh (signed, positive = export from <code>fromba</code>)<br/>"
            "&nbsp;&nbsp;• <code>peak_volatility</code> = mean hourly "
            "standard deviation of interchange during peak<br/><br/>"
            "<b>Normalization</b> (across all pairs):<br/>"
            "&nbsp;&nbsp;• <code>directional_strength = "
            "|peak_flow| / max(|peak_flow|)</code>"
            " — scales 0 to 1, highest-flow pair = 1<br/>"
            "&nbsp;&nbsp;• <code>consistency = 1 − "
            "(peak_volatility / max_volatility)</code>"
            " — scales 0 to 1, lowest-volatility pair = 1<br/><br/>"
            "<b>Composite score</b>:<br/>"
            "&nbsp;&nbsp;<code>signal_score = "
            "100 × (0.6 × directional_strength + 0.4 × consistency)</code><br/><br/>"
            "<b>Interpretation</b>: directional strength weighted higher (60%) "
            "because a large-magnitude persistent flow is a stronger arbitrage "
            "indicator than a small-but-steady one. A route scoring ~80+ "
            "typically represents a structural, non-random congestion signal "
            "worth investigating for cross-market hedging or dispatch arbitrage."
        )

        # Detail table
        st.subheader(
            "Signal Details",
            help=(
                "Tabular view of the top 12 routes with the underlying raw "
                "metrics. <b>Direction</b> reflects the sign of <code>peak_flow</code> "
                "(export if positive, import if negative). <b>Volatility</b> is "
                "the hour-to-hour std-dev of flow during peak hours — lower is "
                "more consistent. <b>Score</b> is the same composite as plotted "
                "above, reproduced here for numerical comparison and CSV export."
            ),
        )
        tbl = disp[
            ["route", "direction", "peak_avg_flow", "peak_volatility", "signal_score"]
        ].copy()
        tbl.columns = ["Route", "Direction", "Avg Flow (MWh)", "Volatility", "Score"]
        st.dataframe(
            tbl.style.format(
                {
                    "Avg Flow (MWh)": "{:,.0f}",
                    "Volatility": "{:,.0f}",
                    "Score": "{:.1f}",
                }
            ).background_gradient(subset=["Score"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )
        _source(
            "EIA Form 930 hourly interchange, aggregated to NERC peak hours "
            "over the full 3-month window."
        )

        _section_divider()

        # Heatmap
        st.subheader(
            "24-Hour Flow Heatmap",
            help=(
                "Diurnal flow signature for the top 8 arbitrage routes, showing "
                "average interchange at every hour of day averaged across the "
                "3-month window. Red cells = export direction, blue = import. "
                "The shaded vertical band marks the NERC peak window "
                "(14:00–20:00). Reading each row horizontally reveals the "
                "daily flow rhythm — a route that alternates red/blue across "
                "the day is behaving differently than one that's solidly one "
                "color, even if their average magnitudes are similar."
            ),
        )
        patterns = compute_interchange_patterns(_D["interchange"])
        if not patterns.empty:
            top_p = signals.head(8)
            rows_h = []
            for _, sr in top_p.iterrows():
                p = patterns[
                    (patterns["fromba"] == sr["fromba"])
                    & (patterns["toba"] == sr["toba"])
                ]
                for _, pp in p.iterrows():
                    rows_h.append(
                        {
                            "Route": f"{sr['fromba']}→{sr['toba']}",
                            "Hour": int(pp["hour"]),
                            "Flow": pp["avg_flow"],
                        }
                    )
            if rows_h:
                hdf = pd.DataFrame(rows_h)
                hpiv = hdf.pivot_table(index="Route", columns="Hour", values="Flow")
                mx = max(abs(hpiv.to_numpy().min()), abs(hpiv.to_numpy().max()), 1)
                fig_h = px.imshow(
                    hpiv,
                    color_continuous_scale="RdBu_r",
                    zmin=-mx,
                    zmax=mx,
                    aspect="auto",
                    labels={"x": "Hour", "color": "MWh"},
                )
                fig_h.add_vrect(
                    x0=NERC_PEAK_START - 0.5,
                    x1=NERC_PEAK_END + 0.5,
                    fillcolor="rgba(0,0,0,0.05)",
                    line_width=2,
                    line_color=C_SLATE,
                    line_dash="dot",
                    annotation_text="NERC Peak",
                    annotation_position="top",
                    annotation_font=dict(size=10, color=C_SLATE),
                )
                fig_h.update_layout(
                    yaxis_title="",
                    coloraxis_colorbar_title="MWh",
                )
                _style(fig_h, h=380)
                st.plotly_chart(fig_h, use_container_width=True)
                _source(
                    "EIA Form 930 hourly interchange for the top 8 arbitrage "
                    "routes, grouped by hour-of-day over the 3-month window."
                )
                _methodology(
                    "For each BA pair <code>(fromba, toba)</code>:<br/>"
                    "&nbsp;&nbsp;1. Filter hourly interchange to this pair.<br/>"
                    "&nbsp;&nbsp;2. Group by <code>hour = period.hour</code> "
                    "(0–23) and compute <code>mean(value)</code> across all "
                    "observations at that hour.<br/>"
                    "&nbsp;&nbsp;3. Resulting 24-cell row is the pair's typical "
                    "diurnal flow shape.<br/><br/>"
                    "Color scale is symmetric: "
                    "<code>zmin = −max(|flow|)</code>, "
                    "<code>zmax = +max(|flow|)</code>, centered at zero. "
                    "This ensures a cell's hue encodes direction (red/blue) "
                    "while its saturation encodes magnitude."
                )

        _section_divider()

        # Route profile
        st.subheader(
            "Route Profile",
            help=(
                "Drill-down view of a single route's 24-hour flow profile. Each "
                "bar shows the average interchange at that hour, with color "
                "indicating direction (blue = export, red = import). Use this "
                "to distinguish between structurally one-way routes (mostly "
                "single color) and bi-directional load-following routes "
                "(alternating colors across the day)."
            ),
        )
        pair_opts = [
            f"{r['fromba']} → {r['toba']}" for _, r in signals.head(10).iterrows()
        ]
        sel_pair = st.selectbox("Select route", pair_opts)
        if sel_pair:
            parts = sel_pair.split(" → ")
            prof = get_pair_hourly_profile(_D["interchange"], parts[0], parts[1])
            if not prof.empty:
                fig_p = go.Figure()
                fig_p.add_vrect(
                    x0=NERC_PEAK_START - 0.5,
                    x1=NERC_PEAK_END + 0.5,
                    fillcolor="rgba(8,145,178,0.08)",
                    line_width=0,
                    annotation_text="Peak",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=C_ACCENT),
                )
                fig_p.add_trace(
                    go.Bar(
                        x=prof["hour"],
                        y=prof["avg_flow"],
                        marker_color=[
                            C_PRIMARY if v > 0 else C_RED for v in prof["avg_flow"]
                        ],
                        marker_line_width=0,
                        hovertemplate="%{x}:00 — %{y:,.0f} MWh<extra></extra>",
                    )
                )
                fig_p.add_hline(y=0, line_color=C_GRID)
                fig_p.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Avg Flow (MWh)",
                    title=sel_pair,
                    bargap=0.15,
                )
                _style(fig_p, h=380)
                st.plotly_chart(fig_p, use_container_width=True)
                _source(
                    f"EIA Form 930 hourly interchange for {sel_pair}, "
                    "grouped by hour-of-day across the 3-month window."
                )
                _methodology(
                    "Each bar is the mean of <code>value</code> for the "
                    f"<code>fromba={parts[0]}, toba={parts[1]}</code> "
                    "interchange series, grouped by hour-of-day (0–23).<br/>"
                    "Bar color: <code>blue</code> if <code>avg_flow > 0</code> "
                    "(export from <code>fromba</code>), <code>red</code> "
                    "otherwise. Signed aggregation preserves direction — if "
                    "this route is truly bidirectional, a simple mean will "
                    "produce a smaller net value than the absolute flows "
                    "you'd see instantaneously."
                )

    # ---------- ISO zonal LMP spreads ----------
    _section_divider()
    st.subheader(
        "ISO-Internal Zonal Spreads",
        help=(
            "Where inter-BA flows (above) capture macro-level arbitrage between "
            "ISOs, zonal spreads capture micro-level arbitrage <i>within</i> a "
            "single ISO. Each row is a price difference between two zones or "
            "hubs inside the same ISO. Persistent spreads during peak hours "
            "indicate transmission constraints binding more often than not — "
            "valuable intel for virtual bidders, congestion revenue rights (CRR) "
            "traders, and storage operators choosing where to site new capacity."
        ),
    )
    spreads = _S["lmp_spreads"]
    if spreads.empty:
        st.info(
            "No LMP zonal spread data — requires `iso_hourly_lmp` table with "
            "≥2 zones per ISO and ≥24h of data."
        )
    else:
        top_spreads = spreads.head(15).copy()
        top_spreads["pair"] = top_spreads["zone_a"] + " ↔ " + top_spreads["zone_b"]

        fig_sp = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            subplot_titles=("Peak-Hour Spread ($/MWh)", "Signal Score"),
            horizontal_spacing=0.08,
        )
        fig_sp.add_trace(
            go.Bar(
                x=top_spreads["peak_spread"],
                y=top_spreads["pair"],
                orientation="h",
                marker_color=C_AMBER,
                showlegend=False,
                text=top_spreads["peak_spread"].round(1),
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig_sp.add_trace(
            go.Bar(
                x=top_spreads["signal_score"],
                y=top_spreads["pair"],
                orientation="h",
                marker_color=C_SECONDARY,
                showlegend=False,
                text=top_spreads["signal_score"].round(1),
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig_sp.update_layout(yaxis=dict(categoryorder="total ascending"))
        _style(fig_sp, h=460)
        st.plotly_chart(fig_sp, use_container_width=True)
        _source(
            "Day-ahead hourly LMP for all CAISO and ERCOT zones/hubs over a "
            "30-day window, retrieved via the <code>gridstatus</code> library."
        )
        _methodology(
            "For every pair of distinct locations <code>(A, B)</code> within "
            "the same ISO:<br/>"
            "&nbsp;&nbsp;1. Compute the hourly spread series "
            "<code>spread<sub>t</sub> = lmp<sub>A,t</sub> − lmp<sub>B,t</sub></code> "
            "(must share a timestamp).<br/>"
            "&nbsp;&nbsp;2. Skip pairs with fewer than 24 hours of overlapping "
            "data to ensure statistical reliability.<br/>"
            "&nbsp;&nbsp;3. Compute: "
            "<code>mean_spread = mean(|spread|)</code>, "
            "<code>peak_spread = mean(|spread|)</code> restricted to NERC peak "
            "hours (14:00–20:00), and "
            "<code>volatility = std(spread)</code>.<br/><br/>"
            "<b>Score</b> (same structure as inter-BA score):<br/>"
            "&nbsp;&nbsp;<code>strength = peak_spread / max(peak_spread)</code><br/>"
            "&nbsp;&nbsp;<code>consistency = 1 − volatility / max(volatility)</code>"
            "<br/>"
            "&nbsp;&nbsp;<code>score = 100 × (0.6 × strength + 0.4 × consistency)</code>"
            "<br/><br/>"
            "Signal score is <i>symmetric</i> — "
            "<code>A ↔ B</code> and <code>B ↔ A</code> are counted once, using "
            "absolute spread. To know which zone is expensive in which hour, "
            "inspect the signed spread in the underlying data."
        )

        spread_tbl = top_spreads[
            [
                "iso",
                "zone_a",
                "zone_b",
                "peak_spread",
                "mean_spread",
                "spread_volatility",
                "signal_score",
            ]
        ].copy()
        spread_tbl.columns = [
            "ISO",
            "Zone A",
            "Zone B",
            "Peak Spread ($)",
            "Mean Spread ($)",
            "Volatility",
            "Score",
        ]
        st.dataframe(
            spread_tbl.style.format(
                {
                    "Peak Spread ($)": "{:,.2f}",
                    "Mean Spread ($)": "{:,.2f}",
                    "Volatility": "{:,.2f}",
                    "Score": "{:.1f}",
                }
            ).background_gradient(subset=["Score"], cmap="Oranges"),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 🌱 TRANSITION SCORING
# ===================================================================
elif page == PAGE_TRANSITION:
    t0 = time.time()
    st.title("🌱 Transition Scoring")
    _cross_ref(
        "Composite scores feed into <b>Executive Briefing</b> and "
        "<b>Compliance Reports §5</b>. High-scoring BAs deserve operational scrutiny."
    )
    st.markdown(
        "Composite scoring of clean energy investment opportunity by region, "
        "combining EIA operational data with the LBNL interconnection queue "
        "and NREL solar resource data.",
        help=(
            "Six factors weighted as: Demand growth (15%) + Renewable headroom (15%) "
            "+ Import dependence (15%) + Fossil transition (15%) + Queue activity (20%) "
            "+ Queue completion rate (20%). When queue data is missing, the four "
            "EIA-based factors fall back to equal 25% weighting."
        ),
    )

    scores = _S["transition"]
    if scores.empty:
        st.info("Insufficient data to compute transition scores.")
    else:
        # Composite ranking
        st.subheader(
            "Composite Ranking",
            help=(
                "BAs ranked by the final 0–100 composite Transition Score, "
                "combining six equally-important analytical dimensions covering "
                "operational pressure (demand growth, import dependence), fuel "
                "mix position (renewable headroom, fossil transition), and "
                "developer validation (queue activity, queue completion). "
                "Higher score means clean-energy investment is likely to be "
                "economically attractive AND physically feasible at this BA."
            ),
        )
        s = scores.sort_values("composite_score")
        fig = go.Figure(
            go.Bar(
                x=s["composite_score"],
                y=s["name"],
                orientation="h",
                marker=dict(
                    color=s["composite_score"],
                    colorscale=[[0, "#d1fae5"], [1, C_GREEN]],
                    line_width=0,
                ),
                text=s["composite_score"].round(1),
                textposition="outside",
                textfont=dict(size=12),
            )
        )
        fig.update_layout(
            xaxis_title="Composite Score (0–100)",
            yaxis_title="",
            showlegend=False,
        )
        _style(fig, h=440)
        st.plotly_chart(fig, use_container_width=True)
        _source(
            "EIA Form 930 (demand, fuel mix, interchange over rolling 3-month "
            "window) + LBNL Queued Up interconnection queue dataset "
            "(<code>queue_ba_summary</code>, thru 2024)."
        )
        _methodology(
            "Each of the six factor scores is computed independently on its "
            "native 0–100 scale (details in individual factor tooltips below), "
            "then combined via weighted sum:<br/><br/>"
            "<code>composite = 0.15 × demand_growth</code><br/>"
            "<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "+ 0.15 × renewable_headroom</code><br/>"
            "<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "+ 0.15 × import_dependence</code><br/>"
            "<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "+ 0.15 × fossil_transition</code><br/>"
            "<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "+ 0.20 × queue_active</code><br/>"
            "<code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
            "+ 0.20 × queue_completion</code><br/><br/>"
            "<b>Weight rationale</b>: the four EIA-derived factors each get "
            "15% because they describe the same underlying thing "
            "(operational conditions) from different angles. The two LBNL "
            "queue factors each get 20% because they reflect "
            "<i>developer-revealed preference</i> — real capital is being "
            "staked, which is the strongest available signal of actual "
            "investment viability.<br/><br/>"
            "<b>Fallback</b>: if LBNL queue data is unavailable for a given "
            "BA, the composite reduces to the four EIA factors at equal 25% "
            "weighting."
        )

        # Radar + table
        col_r, col_t = st.columns([1, 1])
        with col_r:
            st.subheader(
                f"Six-Factor Profile — {sel_ba}",
                help=(
                    "Radar plot decomposes the composite score into its six "
                    "component scores for the selected BA. The shape reveals "
                    "which dimensions drive the composite: a large symmetric "
                    "polygon means balanced opportunity, while a jagged shape "
                    "means the BA scores high on some factors (e.g., renewable "
                    "headroom) but low on others (e.g., queue activity). "
                    "A BA with identical composite scores to another can have "
                    "very different risk/opportunity profiles visible here."
                ),
            )
            ba_s = scores[scores["ba"] == sel_ba]
            if not ba_s.empty:
                cats = [
                    "Demand Growth",
                    "Renewable Headroom",
                    "Import Dependence",
                    "Fossil Transition",
                    "Queue Activity",
                    "Queue Completion",
                ]
                row = ba_s.iloc[0]
                vals = [
                    float(row["demand_growth_score"]),
                    float(row["renewable_headroom_score"]),
                    float(row["import_dependence_score"]),
                    float(row["fossil_transition_score"]),
                    float(row["queue_active_score"])
                    if pd.notna(row["queue_active_score"])
                    else 0.0,
                    float(row["queue_completion_score"])
                    if pd.notna(row["queue_completion_score"])
                    else 0.0,
                ]
                fig_r = go.Figure(
                    go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=cats + [cats[0]],
                        fill="toself",
                        fillcolor="rgba(5,150,105,0.12)",
                        line=dict(color=C_GREEN, width=2),
                    )
                )
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100]),
                        bgcolor="white",
                    ),
                    showlegend=False,
                )
                _style(fig_r, h=400)
                st.plotly_chart(fig_r, use_container_width=True)
                _source(f"EIA Form 930 + LBNL queue, filtered to BA {sel_ba}.")
                _methodology(
                    "<b>1. Demand Growth</b> — compare average daily demand "
                    "between the first and last thirds of the 3-month window: "
                    "<code>growth% = (late − early) / early × 100</code>. "
                    "Score = <code>clamp(50 + 10 × growth%, 0, 100)</code>. "
                    "Neutral BA → 50; growing BAs reward higher.<br/><br/>"
                    "<b>2. Renewable Headroom</b> — current renewable share "
                    "of generation: <code>renew% = sum(gen[SUN,WND,WAT]) / "
                    "sum(gen_all) × 100</code>. "
                    "Score = <code>max(100 − renew%, 0)</code>. "
                    "Low current renewables → high room to add more → high "
                    "score.<br/><br/>"
                    "<b>3. Import Dependence</b> — net interchange position: "
                    "<code>score = clamp(50 − 50 × (net_avg / p95_abs_flow), "
                    "0, 100)</code>. "
                    "Net importers score above 50 (local generation addition "
                    "displaces expensive imports).<br/><br/>"
                    "<b>4. Fossil Transition</b> — fossil share of current "
                    "generation: <code>score = min(fossil% , 100)</code>. "
                    "High fossil % → more fossil to displace → higher "
                    "transition opportunity.<br/><br/>"
                    "<b>5. Queue Activity</b> — active MW in LBNL queue "
                    "normalized across all BAs: <code>score = 100 × "
                    "active_mw / max(active_mw)</code>. Captures developer "
                    "interest in this BA.<br/><br/>"
                    "<b>6. Queue Completion</b> — historical build-out "
                    "success: <code>completion_rate = operational / "
                    "(operational + withdrawn)</code>, then <code>score = "
                    "100 × completion_rate</code>. Captures whether the BA's "
                    "interconnection process actually delivers projects."
                )

        with col_t:
            st.subheader("All Scores", help="Sorted by composite score.")
            display_cols = [
                "ba",
                "name",
                "composite_score",
                "demand_growth_score",
                "renewable_headroom_score",
                "import_dependence_score",
                "fossil_transition_score",
                "queue_active_score",
                "queue_completion_score",
                "current_renewable_pct",
            ]
            num_cols = scores[display_cols].select_dtypes("number").columns
            fmt = {c: "{:.1f}" for c in num_cols}
            st.dataframe(
                scores[display_cols]
                .style.format(fmt, na_rep="—")
                .background_gradient(subset=["composite_score"], cmap="Greens"),
                use_container_width=True,
                hide_index=True,
                height=420,
            )
            _source(
                "Pre-aggregated from EIA Form 930 and LBNL Queued Up dataset. "
                "Full per-factor derivations shown in the radar tooltip to the left."
            )

        _section_divider()

        # LBNL queue panel
        st.subheader(
            "Interconnection Queue — LBNL Project Pipeline",
            help=(
                "Drill-down view of the actual interconnection queue for the "
                "selected BA, sourced from Berkeley Lab's annual Queued Up "
                "report. <b>Active</b> = approved or under study (real capital "
                "at stake); <b>Operational</b> = built and online; "
                "<b>Completion Rate</b> = share of finalized projects that "
                "actually got built rather than withdrawn — a proxy for how "
                "permissive or restrictive the BA's interconnection process is. "
                "High completion rate + high active capacity = investable BA."
            ),
        )
        if _D["queue_ba"].empty:
            st.info("Queue summary data not available.")
        else:
            qb = _D["queue_ba"]
            ba_q = qb[qb["ba"] == sel_ba]
            if ba_q.empty:
                st.info(f"No interconnection queue data tracked for {sel_ba}.")
            else:
                row = ba_q.iloc[0]
                qm1, qm2, qm3, qm4 = st.columns(4)
                qm1.metric(
                    "Active Projects",
                    f"{int(row['active_projects']):,}",
                    help=(
                        "Count of projects with <code>q_status = 'active'</code> "
                        "— projects with signed queue applications still under "
                        "study or awaiting interconnection agreement execution."
                    ),
                )
                qm2.metric(
                    "Active Capacity",
                    f"{row['active_capacity_mw']:,.0f} MW",
                    help=(
                        "Sum of <code>mw1</code> (nameplate capacity at POI) "
                        "across active-status projects. Represents the developer-"
                        "signaled pipeline — not a forecast of actual buildout."
                    ),
                )
                qm3.metric(
                    "Operational",
                    f"{int(row['operational_projects']):,}",
                    help=(
                        "Count of projects with <code>q_status = 'operational'</code> "
                        "— queue entries that graduated to online generation."
                    ),
                )
                comp = row["completion_rate"]
                qm4.metric(
                    "Completion Rate",
                    f"{comp * 100:.1f}%" if pd.notna(comp) else "—",
                    help=(
                        "<code>completion_rate = operational / (operational + "
                        "withdrawn)</code>. Excludes active projects (still in "
                        "progress). Measures historical process success, not "
                        "future conversion probability."
                    ),
                )

            # Resource breakdown for the selected BA
            qb_type = get_queue_breakdown_for_ba(_D["queue_type"], sel_ba)
            if not qb_type.empty:
                st.markdown(f"**Resource type breakdown — {sel_ba}**")
                qb_type_disp = qb_type.head(15).copy()
                fig_qt = go.Figure()
                fig_qt.add_trace(
                    go.Bar(
                        x=qb_type_disp["resource_type"],
                        y=qb_type_disp["active_mw"],
                        name="Active MW",
                        marker_color=C_GREEN,
                    )
                )
                fig_qt.add_trace(
                    go.Bar(
                        x=qb_type_disp["resource_type"],
                        y=qb_type_disp["operational_mw"],
                        name="Operational MW",
                        marker_color=C_PRIMARY,
                    )
                )
                fig_qt.update_layout(
                    barmode="group",
                    xaxis_title="",
                    yaxis_title="MW",
                )
                _style(fig_qt, h=360)
                st.plotly_chart(fig_qt, use_container_width=True)
                _source(
                    f"LBNL <code>queue_type_summary</code> table for BA {sel_ba}, "
                    "aggregated by <code>type_clean</code> resource category."
                )
                _methodology(
                    "For each resource type:<br/>"
                    "&nbsp;&nbsp;• <code>active_mw</code> = "
                    "<code>sum(mw1)</code> for rows where "
                    "<code>q_status = 'active'</code><br/>"
                    "&nbsp;&nbsp;• <code>operational_mw</code> = "
                    "<code>sum(mw1)</code> for rows where "
                    "<code>q_status = 'operational'</code><br/><br/>"
                    "A large Active bar with a small Operational bar signals "
                    "a resource class that's <i>surging now</i> (e.g., battery "
                    "storage in most BAs, or solar+storage hybrids in ERCOT). "
                    "A large Operational bar with small Active signals a "
                    "mature deployment that's tapering off."
                )

            # Cross-BA comparison: total active capacity by BA
            st.markdown("**Cross-BA: active queue capacity**")
            qb_chart = qb[qb["ba"].isin(MAJOR_BA)].sort_values("active_capacity_mw")
            fig_qb = go.Figure(
                go.Bar(
                    x=qb_chart["active_capacity_mw"],
                    y=qb_chart["ba"],
                    orientation="h",
                    marker_color=C_SECONDARY,
                    text=qb_chart["active_capacity_mw"].round(0).astype(int),
                    textposition="outside",
                )
            )
            fig_qb.update_layout(
                xaxis_title="Active Capacity (MW)",
                yaxis_title="",
            )
            _style(fig_qb, h=360)
            st.plotly_chart(fig_qb, use_container_width=True)
            _source(
                "LBNL <code>queue_ba_summary</code> table across the 10 tracked BAs."
            )
            _methodology(
                "Single aggregation per BA: <code>active_capacity_mw = "
                "sum(mw1)</code> restricted to "
                "<code>q_status = 'active'</code>. BA assignment follows a "
                "priority rule (implemented in "
                "<code>data_fetching._assign_ba_from_state_region</code>): "
                "(1) explicit <code>region</code> field if it matches an ISO "
                "(PJM, CAISO, MISO, ERCOT, NYISO, ISO-NE, SPP); (2) state-"
                "based fallback for non-ISO BAs (BPAT, TVA, SOCO). Projects "
                "in states ambiguously covered by multiple BAs use the "
                "primary BA for that state."
            )

        _section_divider()

        # NREL solar resource map
        st.subheader(
            "NREL Solar Resource Map",
            help=(
                "Geographic overlay of solar resource quality (annual Global "
                "Horizontal Irradiance, GHI) at 47 sample locations across the "
                "US. Darker red = higher GHI = better solar siting potential. "
                "Blue diamonds mark the approximate centroids of the 10 tracked "
                "BAs for geographic reference. Use this together with the "
                "Transition Score and Queue data above to match <i>where "
                "resource is good</i> with <i>where interconnection actually "
                "delivers</i>."
            ),
        )
        nrel = _D["nrel"]
        if nrel.empty:
            st.info("NREL resource data not available.")
        else:
            # Map: NREL sample points colored by GHI, plus BA centroids
            fig_m = go.Figure()
            fig_m.add_trace(
                go.Scattergeo(
                    lat=nrel["lat"],
                    lon=nrel["lon"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=nrel["solar_ghi_annual_kwh_m2"],
                        colorscale="YlOrRd",
                        showscale=True,
                        colorbar=dict(title="GHI (kWh/m²)"),
                        line=dict(width=0.5, color="white"),
                    ),
                    text=nrel.apply(
                        lambda r: (
                            f"{r['label']} ({r['state']})<br>"
                            f"GHI: {r['solar_ghi_annual_kwh_m2']:.0f} kWh/m²<br>"
                            f"DNI: {r['solar_dni_annual_kwh_m2']:.0f} kWh/m²"
                        ),
                        axis=1,
                    ),
                    hoverinfo="text",
                    name="NREL solar sites",
                )
            )
            ba_lat = [BA_LATLON[b][0] for b in MAJOR_BA]
            ba_lon = [BA_LATLON[b][1] for b in MAJOR_BA]
            fig_m.add_trace(
                go.Scattergeo(
                    lat=ba_lat,
                    lon=ba_lon,
                    mode="markers+text",
                    marker=dict(size=14, color=C_PRIMARY, symbol="diamond"),
                    text=MAJOR_BA,
                    textposition="top center",
                    textfont=dict(size=10, color=C_PRIMARY),
                    hoverinfo="text",
                    name="Tracked BAs",
                )
            )
            fig_m.update_geos(
                scope="usa",
                showlakes=True,
                lakecolor="#f1f5f9",
                landcolor="#fafafa",
                showland=True,
            )
            fig_m.update_layout(
                geo=dict(bgcolor="white"),
                showlegend=True,
            )
            _style(fig_m, h=500)
            st.plotly_chart(fig_m, use_container_width=True)
            _source(
                "NREL Solar Resource API "
                "(<code>/api/solar/solar_resource/v1.json</code>) at 47 "
                "sampled locations covering all tracked BAs. BA centroids are "
                "analytical approximations, not administrative coordinates."
            )
            _methodology(
                "<b>GHI (Global Horizontal Irradiance)</b>: total solar "
                "radiation received on a horizontal surface, averaged "
                "annually, in <code>kWh/m²/day</code>. Units here are "
                "reported as the annual average daily value. Represents the "
                "theoretical maximum energy a flat-panel PV system could "
                "capture before losses.<br/><br/>"
                "<b>DNI (Direct Normal Irradiance)</b> (shown on hover): "
                "solar radiation on a surface perpendicular to the sun's "
                "rays. Relevant for concentrating solar power (CSP) and "
                "single-axis tracking PV. Higher DNI/GHI ratios indicate "
                "clearer, less diffuse sky conditions.<br/><br/>"
                "<b>Sampling</b>: one NREL API call per sampled point, "
                "parsed from the <code>outputs.avg_ghi.annual</code> and "
                "<code>outputs.avg_dni.annual</code> fields. Points are "
                "chosen to span state centroids and key metros to avoid "
                "sample clustering."
            )

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 📋 COMPLIANCE REPORTS
# ===================================================================
elif page == PAGE_COMPLIANCE:
    t0 = time.time()
    st.title("📋 Compliance Reports")
    _cross_ref(
        "FERC-style summary for the selected BA. <b>§5 Cross-Module Signals</b> "
        "injects the latest anomaly status and transition score."
    )
    st.markdown(
        f"Automated regulatory summary for **{BA_NAMES[sel_ba]}**.",
        help="FERC-style operational summary from EIA Form 930 data. Change BA in sidebar.",
    )

    report = generate_compliance_summary(
        _D["demand"],
        _D["interchange"],
        _D["fuel"],
        sel_ba,
        anomaly_alerts=_S["alerts"],
        transition_scores=_S["transition"],
    )
    sec = report["sections"]

    hc1, hc2 = st.columns([3, 1])
    hc1.markdown(f"## {report['ba']} — {report['ba_name']}")
    if "demand" in sec:
        hc2.caption(
            f"{sec['demand']['period_start'][:10]} to {sec['demand']['period_end'][:10]}"
        )
    _section_divider()

    # 2x2 grid: §1 Demand | §2 Forecast / §3 Interchange | §4 Gen Mix
    c1, c2 = st.columns(2)
    with c1:
        if "demand" in sec:
            d = sec["demand"]
            st.subheader(
                "§1 Demand",
                help=(
                    "Load-side operational summary for the reporting period. "
                    "Peak/min demand bracket the BA's operating envelope; "
                    "average demand anchors capacity-factor calculations and "
                    "is the denominator for MAPE in §2. Hours Reported is a "
                    "data-quality signal — it should be close to "
                    "<code>24 × number_of_days</code>; shortfalls indicate "
                    "reporting gaps on the EIA side."
                ),
            )
            m1, m2 = st.columns(2)
            m1.metric("Avg Demand", f"{d['avg_demand_mwh']:,.0f} MWh")
            m2.metric("Peak Demand", f"{d['peak_demand_mwh']:,.0f} MWh")
            m3, m4 = st.columns(2)
            m3.metric("Min Demand", f"{d['min_demand_mwh']:,.0f} MWh")
            m4.metric("Hours Reported", f"{d['total_hours']:,}")
            _source(
                f"EIA Form 930 <code>type-name = 'Demand'</code> for {sel_ba}, "
                "over the full 3-month rolling window."
            )
            _methodology(
                "<code>avg_demand = mean(value)</code>; "
                "<code>peak_demand = max(value)</code>; "
                "<code>min_demand = min(value)</code>; "
                "<code>total_hours = count(value)</code>, all computed on the "
                "<code>Demand</code> rows of <code>hourly_demand</code>."
            )

    with c2:
        if "forecast_accuracy" in sec:
            fa = sec["forecast_accuracy"]
            st.subheader(
                "§2 Forecast Accuracy",
                help=(
                    "Day-ahead demand forecast performance. <b>MAPE</b> "
                    "(Mean Absolute Percentage Error) is the standard "
                    "industry metric — lower is better, with 1–3% typical "
                    "for mature BAs. <b>MAE</b> (Mean Absolute Error) is "
                    "the MWh magnitude, useful for sizing reserves. "
                    "<b>Bias</b> is the signed mean error: positive means "
                    "the BA systematically under-forecasts demand (actual "
                    "exceeds forecast), negative means it over-forecasts. "
                    "Persistent bias indicates a correctable model issue; "
                    "near-zero bias with high MAE indicates random error."
                ),
            )
            m1, m2 = st.columns(2)
            m1.metric("MAPE", f"{fa['mape']:.2f}%")
            m2.metric("MAE", f"{fa['mae_mwh']:,.0f} MWh")
            m3, m4 = st.columns(2)
            m3.metric("Max Error", f"{fa['max_error_mwh']:,.0f} MWh")
            m4.metric("Bias", f"{fa['bias_mwh']:,.0f} MWh")
            _source(
                "EIA Form 930 <code>Demand</code> and <code>Day-ahead demand "
                f"forecast</code> rows for {sel_ba}, pivoted and differenced."
            )
            _methodology(
                "Let <code>error<sub>t</sub> = demand<sub>t</sub> − "
                "forecast<sub>t</sub></code>. Then:<br/>"
                "&nbsp;&nbsp;• <code>MAPE = mean( |error<sub>t</sub>| / "
                "demand<sub>t</sub> ) × 100</code>  "
                "<i>(hours with demand = 0 excluded)</i><br/>"
                "&nbsp;&nbsp;• <code>MAE = mean( |error<sub>t</sub>| )</code><br/>"
                "&nbsp;&nbsp;• <code>Max Error = max( |error<sub>t</sub>| )</code><br/>"
                "&nbsp;&nbsp;• <code>Bias = mean( error<sub>t</sub> )</code>"
                " — signed, not absolute"
            )

    _section_divider()
    c3, c4 = st.columns(2)
    with c3:
        if "interchange" in sec:
            ix = sec["interchange"]
            st.subheader(
                "§3 Interchange",
                help=(
                    "Cross-border electricity trade. <b>Avg Net</b> "
                    "characterizes the BA's structural role: positive = net "
                    "exporter over the period, negative = net importer. "
                    "<b>Partners</b> counts how many distinct BAs this BA "
                    "exchanges power with — a proxy for transmission "
                    "connectivity and market optionality. <b>Peak Export / "
                    "Import</b> give the 24-hour envelope of directional "
                    "flow extremes."
                ),
            )
            m1, m2 = st.columns(2)
            m1.metric("Avg Net", f"{ix['avg_net_mwh']:,.0f} MWh")
            m2.metric("Partners", f"{ix['n_trading_partners']}")
            m3, m4 = st.columns(2)
            m3.metric("Peak Export", f"{ix['peak_export_mwh']:,.0f} MWh")
            m4.metric("Peak Import", f"{ix['peak_import_mwh']:,.0f} MWh")
            _source(
                "EIA Form 930 <code>hourly_interchange</code> rows where "
                f"<code>fromba = {sel_ba}</code>."
            )
            _methodology(
                "EIA convention: positive <code>value</code> = export, "
                "negative = import.<br/>"
                "&nbsp;&nbsp;• <code>avg_net = mean(value)</code><br/>"
                "&nbsp;&nbsp;• <code>peak_export = max(value)</code> "
                "(most positive single hour)<br/>"
                "&nbsp;&nbsp;• <code>peak_import = min(value)</code> "
                "(most negative single hour)<br/>"
                "&nbsp;&nbsp;• <code>n_trading_partners = "
                "nunique(toba)</code> — count of distinct destinations"
            )

    with c4:
        if "generation_mix" in sec:
            gm = sec["generation_mix"]
            st.subheader(
                "§4 Generation Mix",
                help=(
                    "Share of total generation by fuel type over the reporting "
                    "period. Ordered by share magnitude. High gas or coal share "
                    "implies sensitivity to fuel-price volatility (§ Henry Hub "
                    "price) and higher emissions intensity. Rising solar/wind "
                    "share reflects installed-capacity transitions visible over "
                    "the 3-month window; material shifts in mix are rare over "
                    "short windows and usually indicate seasonality."
                ),
            )
            if gm["fuel_shares_pct"]:
                mix_df = pd.DataFrame(
                    [
                        {"Fuel": FUEL_LABELS.get(k, k), "Share": v}
                        for k, v in gm["fuel_shares_pct"].items()
                    ]
                ).sort_values("Share", ascending=True)
                fig = go.Figure(
                    go.Bar(
                        x=mix_df["Share"],
                        y=mix_df["Fuel"],
                        orientation="h",
                        marker_color=[
                            FUEL_COLORS.get(f, "#94a3b8") for f in mix_df["Fuel"]
                        ],
                        text=[f"{v:.1f}%" for v in mix_df["Share"]],
                        textposition="outside",
                        textfont=dict(size=11),
                    )
                )
                fig.update_layout(
                    xaxis_title="Share (%)",
                    yaxis_title="",
                    showlegend=False,
                )
                _style(fig, h=280)
                st.plotly_chart(fig, use_container_width=True)
                _source(
                    f"EIA Form 930 <code>hourly_fuel_type</code> rows for "
                    f"{sel_ba} aggregated over the full 3-month window."
                )
                _methodology(
                    "For each fuel type <code>f</code>:<br/>"
                    "&nbsp;&nbsp;<code>share<sub>f</sub> = sum(value | "
                    "fueltype = f) / sum(value<sub>all</sub>) × 100</code>"
                    "<br/><br/>"
                    "Shares sum to 100% by construction. EIA reports "
                    "hourly values as MW (instantaneous generation); "
                    "aggregation by sum treats each hour as a 1 MWh bucket, "
                    "which is the correct energy-weighted interpretation."
                )

    _section_divider()

    # §5 Cross-Module Signals
    st.subheader(
        "§5 Cross-Module Signals",
        help=(
            "Novel addition not present in traditional compliance reports: "
            "surfaces the latest derived signals from other analytical "
            "modules in this platform alongside the regulatory metrics above. "
            "Gives the reader a single view bridging operational reporting "
            "(§1–§4) with forward-looking analytical perspectives (anomaly "
            "status from monitoring, composite transition score, active "
            "interconnection pipeline). Particularly useful for regulators "
            "and internal risk teams who need context beyond point-in-time "
            "compliance."
        ),
    )
    cross = sec.get("cross_module_signals")
    if not cross:
        st.info("No cross-module signals available for this BA.")
    else:
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.markdown("**Anomaly Status**")
            if "anomaly_status" in cross:
                st.markdown(
                    _status_pill(cross["anomaly_status"]), unsafe_allow_html=True
                )
                if "hours_above_p95" in cross:
                    st.caption(
                        f"Hours above P95 in last 6h: {cross['hours_above_p95']}"
                    )
            else:
                st.caption("No anomaly data.")
        with cm2:
            if "transition_composite_score" in cross:
                cm2.metric(
                    "Transition Score", f"{cross['transition_composite_score']:.1f}"
                )
        with cm3:
            if "active_queue_mw" in cross:
                cm3.metric("Active Queue", f"{cross['active_queue_mw']:,.0f} MW")
        _source(
            "Injected from the Anomaly Detection and Transition Scoring "
            "modules for this BA. Click through to those modules for full "
            "derivation."
        )
        _methodology(
            "<b>Anomaly Status</b>: the <code>status</code> field from the "
            "anomaly alerts table (<code>RED</code>, <code>YELLOW</code>, or "
            "<code>NORMAL</code>). See the Anomaly Detection module for the "
            "P90/P95 derivation.<br/><br/>"
            "<b>Transition Score</b>: the <code>composite_score</code> from "
            "the six-factor Transition Scoring model. See the Transition "
            "Scoring module for the full weighting and factor derivations."
            "<br/><br/>"
            "<b>Active Queue (MW)</b>: the <code>active_capacity_mw</code> "
            "from the LBNL queue summary. See Transition Scoring for methodology."
        )

    _section_divider()
    st.caption(
        "Auto-generated from EIA Form 930 data. Verify against primary sources for filings."
    )
    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 📄 ABOUT
# ===================================================================
elif page == PAGE_ABOUT:
    t0 = time.time()
    st.title("📄 About")

    st.markdown(
        """
        # ⚡ Grid Intelligence Platform

        **Bloomberg Terminal for those who can't afford one.**

        An open-source power market intelligence platform that turns public EIA,
        NREL, and LBNL data into actionable decision signals for small retail
        electricity providers, virtual power plant operators, and clean energy
        developers.

        ---
        ## Mission

        Professional-grade electricity market intelligence has historically been
        gated behind five- and six-figure subscriptions. Yet the underlying data
        — demand, forecasts, generation mix, interchange, interconnection queues,
        and resource potential — is either public or available at low cost.

        This platform consolidates those public sources into a single, decision-
        oriented dashboard for the operators and developers who can't afford
        proprietary tools but still need to make capital-allocation, dispatch,
        and siting decisions every week.

        ---
        ## Platform modules

        The platform analyzes **10 major U.S. balancing authorities (BAs)**:
        CISO, ERCO, PJM, MISO, NYIS, ISNE, SWPP, SOCO, TVA, BPAT.

        It is organized around six modules:

        1. **📊 Executive Briefing** — single-pane snapshot of one BA, drawing
           signals from every other module
        2. **🚨 Anomaly Detection** — BA-level forecast error monitoring with
           P90/P95 control charts and cross-BA distribution comparison
        3. **💰 Arbitrage Signals** — persistent peak-hour interchange flow
           patterns suggesting cross-market price imbalance
        4. **🌱 Transition Scoring** — six-factor renewable opportunity scoring
           combining EIA operational data, LBNL queue activity, and NREL solar
           resource potential
        5. **📋 Compliance Reports** — FERC-style §1–§4 operational summaries
           with §5 cross-module signal injection
        6. **📄 About** — this page

        ---
        ## Methodology

        ### Forecast error monitoring
        For each BA we compute the absolute hourly forecast error and its P90
        and P95 thresholds over the rolling window. A BA is flagged **RED** if
        ≥3 of the last 6 hours exceeded P95, **YELLOW** if ≥3 exceeded P90,
        otherwise **NORMAL**.

        ### Arbitrage signal scoring
        For every BA-pair, we compute average hourly interchange and isolate
        NERC peak hours (14:00–20:00). The signal score combines:

        - **Directional strength (60%)** — peak-hour absolute flow magnitude,
          normalized across all pairs
        - **Consistency (40%)** — inverse of flow volatility

        ### Transition scoring (six-factor composite)
        Each BA receives a 0–100 score on six factors:

        | Factor | Weight | Source |
        |---|---|---|
        | Demand growth | 15% | EIA hourly demand |
        | Renewable headroom | 15% | EIA fuel-type generation |
        | Import dependence | 15% | EIA interchange |
        | Fossil transition opportunity | 15% | EIA fuel-type generation |
        | Queue activity | 20% | LBNL interconnection queue |
        | Queue completion rate | 20% | LBNL interconnection queue |

        When LBNL queue data is unavailable, the four EIA factors fall back to
        equal 25% weighting.

        ### Compliance §5 cross-module injection
        The compliance report goes beyond operational metrics by injecting the
        live anomaly status and transition score into §5, giving regulators and
        operators a consolidated regulatory + strategic snapshot in one view.

        ---
        ## Data sources

        ### EIA — Energy Information Administration
        - **Demand & forecast** (`electricity/rto/region-data`, hourly)
        - **Interchange** (`electricity/rto/interchange-data`, hourly)
        - **Generation by fuel type** (`electricity/rto/fuel-type-data`, hourly)
        - **Henry Hub natural gas price** (`natural-gas/pri/fut`, daily)

        ### Open-Meteo
        - Daily temperature data for representative BA locations

        ### NREL — National Renewable Energy Laboratory
        - Solar GHI / DNI annual averages at 47 sample sites across the US,
          fetched via the NREL Solar Resource API

        ### gridstatus — ISO LMPs
        - Day-ahead hourly locational marginal prices for CAISO and ERCOT
          zones/hubs. Optional PJM coverage when `PJM_API_KEY` is configured.

        ### LBNL — Lawrence Berkeley National Laboratory
        - Interconnection queue dataset (36k+ project-level rows) with status,
          capacity, resource type, and BA mapping

        Fuel categories: NG (Natural Gas), SUN (Solar), WND (Wind), NUC (Nuclear),
        COL (Coal), WAT (Hydro), OIL (Oil), OTH (Other).

        ---
        ## Architecture

        **External APIs → GitHub Actions ETL → BigQuery → Streamlit dashboard**

        ### Stage 1: ETL pipeline
        `load_to_bigquery.py` runs daily at 09:00 UTC via GitHub Actions cron.
        Each source is fetched independently with isolated failure handling so
        a single source outage doesn't kill the rest of the pipeline. After
        raw writes, pre-aggregated summary tables are built directly in
        BigQuery via SQL.

        ### Stage 2: Dashboard loading
        The Streamlit app loads all major datasets once at startup using
        `st.cache_resource(ttl=3600)`, then serves all pages from in-memory
        DataFrames for sub-second navigation across modules.

        ### BigQuery dataset: `sipa-adv-c-silly-penguin.eia_data`
        **Raw tables**: `hourly_demand`, `hourly_interchange`, `hourly_fuel_type`,
        `daily_ng_price`, `daily_weather`, `nrel_resource_locations`,
        `interconnection_queue`.

        **Aggregated tables**: `daily_demand_summary`, `daily_interchange_summary`,
        `daily_fuel_summary`, `ba_mape_ranking`, `queue_ba_summary`,
        `queue_type_summary`.

        ---
        ## Limitations

        We are committed to honesty about what this platform can and cannot do:

        - **Partial LMP coverage.** ISO LMP integration via the `gridstatus`
          library covers **CAISO and ERCOT** zones/hubs (day-ahead hourly).
          PJM is supported but requires a free Data Miner 2 API key; MISO,
          NYISO, ISONE, and SPP are not currently included. Spike, negative-
          price, and zonal-spread analytics run on whatever ISOs are present.
        - **EIA data is operational, not market.** EIA Form 930 reports
          physical demand and generation, not bid stacks, capacity prices, or
          ancillary service markets.
        - **Weather is single-point.** One reference location per BA — adequate
          for high-level demand-temperature correlation, not sufficient for
          fine-grained generation forecasting.
        - **Arbitrage signals reflect physical flow patterns, not confirmed
          price spreads.** Without LMP data, we infer opportunity from
          persistent directional flows during peak hours rather than measuring
          spreads directly.
        - **Rolling 3-month window.** The ETL pulls a rolling 3-month window
          and full-replaces. Suitable for short-horizon operational analysis;
          not for long-term structural inference.
        - **Queue data is not capacity-firm.** LBNL queue rows include
          speculative early-stage projects. Active capacity figures should be
          read as *interest signal*, not as a forecast of actual buildout.

        ---
        ## Tech stack

        - **Frontend:** Streamlit, Plotly
        - **ETL:** Python 3.11, GitHub Actions
        - **Data warehouse:** Google BigQuery
        - **Validation:** Pandera schemas
        - **Lint/format:** ruff
        - **Deployment:** Streamlit Cloud

        ---
        ## Team

        **Xingyi Wang & Wuhao Xia**
        Columbia SIPA — *Advanced Computing for Policy*

        Repository: `advanced-computing/silly-penguin`
        """
    )

    st.caption(f"Loaded in {time.time() - t0:.2f}s")
