"""Grid Intelligence Platform — Open-Source Power Market Intelligence."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import (
    build_geographic_summary,
    calculate_grid_kpis,
    compute_generation_mix,
    compute_interchange_patterns,
    compute_net_interchange,
    compute_renewable_siting_scores,
    detect_anomalies,
    generate_compliance_summary,
    get_ba_error_distribution,
    get_pair_hourly_profile,
    identify_arbitrage_opportunities,
    prepare_demand_pivot,
)

# ==========================================
# Config & Theme
# ==========================================
st.set_page_config(
    page_title="Grid Intelligence",
    page_icon="⚡",
    layout="wide",
)

C_PRIMARY = "#1e40af"
C_SECONDARY = "#0369a1"
C_ACCENT = "#0891b2"
C_RED = "#dc2626"
C_AMBER = "#d97706"
C_GREEN = "#059669"
C_SLATE = "#475569"
C_GRID = "#e2e8f0"

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
NERC_PEAK_START = 14
NERC_PEAK_END = 20


def _style(fig, h=440):
    """Professional styling for all Plotly figures."""
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


# ==========================================
# Auth & Data
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
_DS = f"{GCP_PROJECT}.{BQ_DATASET}"


@st.cache_resource(show_spinner="⚡ Loading market data...")
def _load_all():
    from google.cloud import bigquery

    cl = bigquery.Client(project=GCP_PROJECT, credentials=_creds)

    def q(s):
        return cl.query(s).to_dataframe()

    d = {}

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

    df = q(f"SELECT date, CAST(ng_price AS FLOAT64) AS ng_price FROM `{_DS}.daily_ng_price`")
    df["date"] = pd.to_datetime(df["date"])
    d["ng_price"] = df

    df = q(
        "SELECT date, ba, "
        "CAST(avg_temp AS FLOAT64) AS avg_temp, "
        "CAST(max_temp AS FLOAT64) AS max_temp, "
        "CAST(min_temp AS FLOAT64) AS min_temp "
        f"FROM `{_DS}.daily_weather`"
    )
    df["date"] = pd.to_datetime(df["date"])
    d["weather"] = df

    d["ranking"] = q(f"SELECT * FROM `{_DS}.ba_mape_ranking` ORDER BY mape")
    return d


_D = _load_all()

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("⚡ Grid Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Module",
    [
        "🚨 Anomaly Detection",
        "💰 Arbitrage Signals",
        "🌱 Renewable Siting",
        "📋 Compliance Reports",
        "📊 Market Overview",
        "📄 About",
    ],
)
st.sidebar.markdown("---")
sel_ba = st.sidebar.selectbox(
    "Balancing Authority",
    MAJOR_BA,
    format_func=lambda x: f"{x} — {BA_NAMES[x]}",
)
days = st.sidebar.slider("Time window (days)", 1, 30, 7)


# ===================================================================
# 🚨 ANOMALY DETECTION
# ===================================================================
if page == "🚨 Anomaly Detection":
    t0 = time.time()
    st.title("🚨 Anomaly Detection")
    st.markdown(
        "Monitors forecast errors across all balancing "
        "authorities and flags regions where recent errors "
        "persistently exceed historical norms.",
        help=(
            "For each BA we compute the P90 and P95 of "
            "historical absolute forecast error. If ≥3 of "
            "the last 6 hours exceed P95 → RED. "
            "≥3 exceed P90 → YELLOW. Otherwise GREEN."
        ),
    )
    alerts = detect_anomalies(_D["demand"])
    if alerts.empty:
        st.info("No data available.")
    else:
        red = alerts[alerts["status"] == "RED"]
        yel = alerts[alerts["status"] == "YELLOW"]
        grn = alerts[alerts["status"] == "NORMAL"]

        col_r, col_y, col_g = st.columns(3)
        with col_r:
            st.markdown(f"#### 🔴 Critical ({len(red)})")
            for _, r in red.iterrows():
                st.error(
                    f"**{r['ba']}** — "
                    f"{BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: **{r['latest_error']:,.0f}** MWh"
                    f" · 6h avg: {r['recent_mean_error']:,.0f}"
                    f" · P95: {r['p95_threshold']:,.0f}",
                    icon="🔴",
                )
        with col_y:
            st.markdown(f"#### 🟡 Warning ({len(yel)})")
            for _, r in yel.iterrows():
                st.warning(
                    f"**{r['ba']}** — "
                    f"{BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: **{r['latest_error']:,.0f}** MWh"
                    f" · 6h avg: {r['recent_mean_error']:,.0f}"
                    f" · P90: {r['p90_threshold']:,.0f}",
                    icon="🟡",
                )
        with col_g:
            st.markdown(f"#### 🟢 Normal ({len(grn)})")
            for _, r in grn.iterrows():
                st.success(
                    f"**{r['ba']}** — "
                    f"{BA_NAMES.get(r['ba'], '')}  \n"
                    f"Error: {r['latest_error']:,.0f} MWh"
                    " · Within normal range",
                    icon="🟢",
                )

        st.markdown("---")

        # Control chart
        st.subheader(
            f"{sel_ba} — Error Control Chart",
            help=(
                "A control chart plots the metric with "
                "upper control limits (P90, P95). Points "
                "outside = 'out of control'."
            ),
        )
        ba_err = get_ba_error_distribution(_D["demand"], sel_ba)
        if not ba_err.empty:
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
                    annotation_text=f"P95 = {p95:,.0f}",
                    annotation_font=dict(color=C_RED, size=11),
                )
            if p90:
                fig.add_hline(
                    y=p90,
                    line_dash="dot",
                    line_color=C_AMBER,
                    line_width=1.2,
                    annotation_text=f"P90 = {p90:,.0f}",
                    annotation_font=dict(color=C_AMBER, size=11),
                    annotation_position="bottom right",
                )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Absolute Error (MWh)",
            )
            _style(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Violin plot
        st.subheader(
            "Cross-BA Error Comparison",
            help=(
                "Violin plots show both distribution shape "
                "and density of forecast errors for each BA."
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
            fig_v = px.violin(
                err_df,
                x="BA",
                y="abs_error",
                box=True,
                points=False,
                color_discrete_sequence=[C_PRIMARY],
                labels={
                    "abs_error": "Absolute Error (MWh)",
                    "BA": "",
                },
            )
            _style(fig_v, h=400)
            st.plotly_chart(fig_v, use_container_width=True)

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 💰 ARBITRAGE SIGNALS
# ===================================================================
elif page == "💰 Arbitrage Signals":
    t0 = time.time()
    st.title("💰 Arbitrage Signals")
    st.markdown(
        "Identifies persistent directional power flows "
        "between regions during peak hours, indicating "
        "potential price differentials.",
        help=(
            "We compute average hourly interchange for "
            "every BA pair, focusing on NERC peak hours "
            "(14:00–20:00). Signal = 60% directional "
            "strength + 40% consistency."
        ),
    )
    signals = identify_arbitrage_opportunities(_D["interchange"])
    if signals.empty:
        st.info("No data.")
    else:
        disp = signals.head(12).copy()
        disp["route"] = disp["fromba"] + " → " + disp["toba"]

        st.subheader(
            "Top Opportunities — NERC Peak (14–20h)",
            help=("Routes ranked by signal strength. High score = large, consistent flow."),
        )
        top8 = disp.head(8).sort_values("signal_score")
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=True,
            subplot_titles=(
                "Signal Score",
                "Peak Avg Flow (MWh)",
                "Consistency",
            ),
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
        cons = top8.get(
            "consistency",
            pd.Series([0.5] * len(top8)),
        )
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

        # Table
        st.subheader(
            "Signal Details",
            help=("Direction: net flow direction. Volatility: lower = more consistent."),
        )
        tbl = disp[
            [
                "route",
                "direction",
                "peak_avg_flow",
                "peak_volatility",
                "signal_score",
            ]
        ].copy()
        tbl.columns = [
            "Route",
            "Direction",
            "Avg Flow (MWh)",
            "Volatility",
            "Score",
        ]
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

        st.markdown("---")

        # Heatmap
        st.subheader(
            "24-Hour Flow Heatmap",
            help=("Red = export, Blue = import. Shaded band = NERC peak hours."),
        )
        patterns = compute_interchange_patterns(_D["interchange"])
        if not patterns.empty:
            top_p = signals.head(8)
            rows_h = []
            for _, sr in top_p.iterrows():
                p = patterns[
                    (patterns["fromba"] == sr["fromba"]) & (patterns["toba"] == sr["toba"])
                ]
                for _, pp in p.iterrows():
                    rows_h.append(
                        {
                            "Route": (f"{sr['fromba']}→{sr['toba']}"),
                            "Hour": int(pp["hour"]),
                            "Flow": pp["avg_flow"],
                        }
                    )
            if rows_h:
                hdf = pd.DataFrame(rows_h)
                hpiv = hdf.pivot_table(
                    index="Route",
                    columns="Hour",
                    values="Flow",
                )
                mx = max(
                    abs(hpiv.to_numpy().min()),
                    abs(hpiv.to_numpy().max()),
                    1,
                )
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

        st.markdown("---")

        # Route detail
        st.subheader(
            "Route Profile",
            help="Shaded region = NERC peak hours.",
        )
        pair_opts = [f"{r['fromba']} → {r['toba']}" for _, r in signals.head(10).iterrows()]
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
                        marker_color=[C_PRIMARY if v > 0 else C_RED for v in prof["avg_flow"]],
                        marker_line_width=0,
                        hovertemplate=("%{x}:00 — %{y:,.0f} MWh<extra></extra>"),
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

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 🌱 RENEWABLE SITING
# ===================================================================
elif page == "🌱 Renewable Siting":
    t0 = time.time()
    st.title("🌱 Renewable Investment Scoring")
    st.markdown(
        "Composite scoring of clean energy investment opportunity by region.",
        help=(
            "Each BA scored 0–100 on four equally weighted "
            "factors: (1) Demand growth, (2) Renewable "
            "headroom, (3) Import dependence, "
            "(4) Fossil transition opportunity."
        ),
    )
    scores = compute_renewable_siting_scores(_D["demand"], _D["fuel"], _D["interchange"])
    if scores.empty:
        st.info("Insufficient data.")
    else:
        st.subheader(
            "Composite Ranking",
            help="Higher score = greater opportunity.",
        )
        s = scores.sort_values("composite_score")
        fig = go.Figure(
            go.Bar(
                x=s["composite_score"],
                y=s["name"],
                orientation="h",
                marker=dict(
                    color=s["composite_score"],
                    colorscale=[
                        [0, "#d1fae5"],
                        [1, C_GREEN],
                    ],
                    line_width=0,
                ),
                text=s["composite_score"].round(1),
                textposition="outside",
                textfont=dict(size=12),
            )
        )
        fig.update_layout(
            xaxis_title="Score (0–100)",
            yaxis_title="",
            showlegend=False,
        )
        _style(fig, h=440)
        st.plotly_chart(fig, use_container_width=True)

        # Radar + Table side by side
        col_r, col_t = st.columns([1, 1])
        with col_r:
            st.subheader(
                f"Score Profile — {sel_ba}",
                help=("Radar shows four scoring dimensions. Larger shape = stronger opportunity."),
            )
            ba_s = scores[scores["ba"] == sel_ba]
            if not ba_s.empty:
                cats = [
                    "Demand Growth",
                    "Renewable Headroom",
                    "Import Dependence",
                    "Fossil Transition",
                ]
                vals = [
                    ba_s["demand_growth_score"].iloc[0],
                    ba_s["renewable_headroom_score"].iloc[0],
                    ba_s["import_dependence_score"].iloc[0],
                    ba_s["fossil_transition_score"].iloc[0],
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
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                        ),
                        bgcolor="white",
                    ),
                    showlegend=False,
                )
                _style(fig_r, h=380)
                st.plotly_chart(fig_r, use_container_width=True)

        with col_t:
            st.subheader("All Scores")
            num_cols = scores.select_dtypes("number").columns
            fmt = {c: "{:.1f}" for c in num_cols}
            st.dataframe(
                scores[
                    [
                        "ba",
                        "name",
                        "composite_score",
                        "demand_growth_score",
                        "renewable_headroom_score",
                        "import_dependence_score",
                        "fossil_transition_score",
                        "current_renewable_pct",
                    ]
                ]
                .style.format(fmt)
                .background_gradient(
                    subset=["composite_score"],
                    cmap="Greens",
                ),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        # Map
        st.subheader(
            "Regional Map",
            help=("Bubble size and colour represent the composite score."),
        )
        geo = build_geographic_summary(_D["demand"], _D["interchange"], _D["fuel"])
        if not geo.empty:
            geo = geo.merge(
                scores[["ba", "composite_score"]],
                on="ba",
                how="left",
            )
            gp = geo.dropna(subset=["composite_score"])
            fig_m = px.scatter_geo(
                gp,
                lat="lat",
                lon="lon",
                size="composite_score",
                color="composite_score",
                hover_name="name",
                color_continuous_scale=[
                    [0, "#d1fae5"],
                    [0.5, "#34d399"],
                    [1, "#065f46"],
                ],
                size_max=30,
                hover_data={
                    "lat": False,
                    "lon": False,
                    "composite_score": ":.1f",
                },
            )
            fig_m.update_geos(
                scope="usa",
                showlakes=True,
                lakecolor="#f1f5f9",
                landcolor="#fafafa",
                showland=True,
            )
            fig_m.update_layout(
                coloraxis_colorbar_title="Score",
                geo=dict(bgcolor="white"),
            )
            _style(fig_m, h=460)
            st.plotly_chart(fig_m, use_container_width=True)

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 📋 COMPLIANCE REPORTS
# ===================================================================
elif page == "📋 Compliance Reports":
    t0 = time.time()
    st.title("📋 Compliance Reports")
    st.markdown(
        f"Automated regulatory summary for **{BA_NAMES[sel_ba]}**.",
        help=("FERC-style operational summary from EIA Form 930 data. Change BA in sidebar."),
    )
    report = generate_compliance_summary(
        _D["demand"],
        _D["interchange"],
        _D["fuel"],
        sel_ba,
    )
    sec = report["sections"]

    hc1, hc2 = st.columns([3, 1])
    hc1.markdown(f"## {report['ba']} — {report['ba_name']}")
    if "demand" in sec:
        hc2.caption(f"{sec['demand']['period_start'][:10]} to {sec['demand']['period_end'][:10]}")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        if "demand" in sec:
            d = sec["demand"]
            st.subheader(
                "§1 Demand",
                help="Hourly demand statistics.",
            )
            m1, m2 = st.columns(2)
            m1.metric(
                "Avg Demand",
                f"{d['avg_demand_mwh']:,.0f} MWh",
            )
            m2.metric(
                "Peak Demand",
                f"{d['peak_demand_mwh']:,.0f} MWh",
            )
            m3, m4 = st.columns(2)
            m3.metric(
                "Min Demand",
                f"{d['min_demand_mwh']:,.0f} MWh",
            )
            m4.metric("Hours Reported", f"{d['total_hours']:,}")

    with c2:
        if "forecast_accuracy" in sec:
            fa = sec["forecast_accuracy"]
            st.subheader(
                "§2 Forecast Accuracy",
                help=("MAPE: lower = better. Bias: positive = demand > forecast."),
            )
            m1, m2 = st.columns(2)
            m1.metric("MAPE", f"{fa['mape']:.2f}%")
            m2.metric("MAE", f"{fa['mae_mwh']:,.0f} MWh")
            m3, m4 = st.columns(2)
            m3.metric(
                "Max Error",
                f"{fa['max_error_mwh']:,.0f} MWh",
            )
            m4.metric("Bias", f"{fa['bias_mwh']:,.0f} MWh")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        if "interchange" in sec:
            ix = sec["interchange"]
            st.subheader(
                "§3 Interchange",
                help=("Positive = net export. Negative = net import."),
            )
            m1, m2 = st.columns(2)
            m1.metric("Avg Net", f"{ix['avg_net_mwh']:,.0f} MWh")
            m2.metric("Partners", f"{ix['n_trading_partners']}")
            m3, m4 = st.columns(2)
            m3.metric(
                "Peak Export",
                f"{ix['peak_export_mwh']:,.0f} MWh",
            )
            m4.metric(
                "Peak Import",
                f"{ix['peak_import_mwh']:,.0f} MWh",
            )

    with c4:
        if "generation_mix" in sec:
            gm = sec["generation_mix"]
            st.subheader(
                "§4 Generation Mix",
                help="Fuel shares in total generation.",
            )
            if gm["fuel_shares_pct"]:
                mix_df = pd.DataFrame(
                    [
                        {
                            "Fuel": FUEL_LABELS.get(k, k),
                            "Share": v,
                        }
                        for k, v in gm["fuel_shares_pct"].items()
                    ]
                ).sort_values("Share", ascending=True)
                fig = go.Figure(
                    go.Bar(
                        x=mix_df["Share"],
                        y=mix_df["Fuel"],
                        orientation="h",
                        marker_color=[FUEL_COLORS.get(f, "#94a3b8") for f in mix_df["Fuel"]],
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

    st.markdown("---")
    st.caption("Auto-generated from EIA Form 930 data. Verify against primary sources for filings.")
    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 📊 MARKET OVERVIEW
# ===================================================================
elif page == "📊 Market Overview":
    t0 = time.time()
    st.title("📊 Market Overview")
    st.markdown(
        f"Operational snapshot for **{BA_NAMES[sel_ba]}**.",
        help=("Demand, forecast accuracy, generation mix, interchange, and energy prices."),
    )
    pivot = prepare_demand_pivot(_D["demand"], sel_ba)
    actual, forecast, delta = calculate_grid_kpis(pivot)

    if actual is not None:
        kc = st.columns(4)
        kc[0].metric("Latest Demand", f"{actual:,.0f} MWh")
        kc[1].metric("Forecast", f"{forecast:,.0f} MWh")
        kc[2].metric(
            "Error",
            f"{delta:,.0f} MWh",
            delta_color="inverse",
        )
        rk = _D["ranking"]
        br = rk[rk["ba"] == sel_ba] if not rk.empty else pd.DataFrame()
        if not br.empty:
            kc[3].metric("MAPE", f"{br['mape'].iloc[0]:.2f}%")
    st.markdown("---")

    # Demand + Error subplot
    st.subheader(
        f"Demand & Forecast — Last {days} Days",
        help=("Top: actual vs forecast. Bottom: hourly error bars."),
    )
    if not pivot.empty:
        dd = pivot.head(days * 24).sort_index()
        cols_ok = {
            "Demand",
            "Day-ahead demand forecast",
        }.issubset(dd.columns)
        if cols_ok:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.65, 0.35],
                vertical_spacing=0.04,
            )
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd["Demand"],
                    name="Actual",
                    line=dict(color=C_PRIMARY, width=2),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd["Day-ahead demand forecast"],
                    name="Forecast",
                    line=dict(
                        color=C_AMBER,
                        width=2,
                        dash="dash",
                    ),
                ),
                row=1,
                col=1,
            )
            if "Forecast Error" in dd.columns:
                fig.add_trace(
                    go.Bar(
                        x=dd.index,
                        y=dd["Forecast Error"],
                        name="Error",
                        marker_color=np.where(
                            dd["Forecast Error"] > 0,
                            "rgba(220,38,38,0.5)",
                            "rgba(30,64,175,0.5)",
                        ),
                    ),
                    row=2,
                    col=1,
                )
                fig.add_hline(
                    y=0,
                    line_color=C_GRID,
                    row=2,
                    col=1,
                )
            fig.update_yaxes(title_text="MWh", row=1, col=1)
            fig.update_yaxes(title_text="Error", row=2, col=1)
            _style(fig, h=520)
            st.plotly_chart(fig, use_container_width=True)

    # Gen mix + Interchange
    cg, ci = st.columns(2)
    with cg:
        st.subheader(
            "Generation Mix",
            help="Daily average generation by fuel.",
        )
        mix = compute_generation_mix(_D["fuel"], sel_ba)
        if not mix.empty:
            mc = mix.copy()
            mc["Fuel"] = mc["fueltype"].map(FUEL_LABELS)
            fig2 = px.area(
                mc,
                x="date",
                y="avg_generation_mwh",
                color="Fuel",
                labels={
                    "avg_generation_mwh": "MWh",
                    "date": "",
                },
                color_discrete_map=FUEL_COLORS,
            )
            _style(fig2, h=380)
            st.plotly_chart(fig2, use_container_width=True)

    with ci:
        st.subheader(
            "Net Interchange",
            help=("Positive = export. Negative = import from neighbors."),
        )
        net = compute_net_interchange(_D["interchange"], sel_ba)
        if not net.empty:
            rec = net.tail(days * 24)
            fig3 = go.Figure(
                go.Scatter(
                    x=rec["period"],
                    y=rec["net_interchange_mwh"],
                    fill="tozeroy",
                    fillcolor="rgba(8,145,178,0.08)",
                    line=dict(color=C_ACCENT, width=1.5),
                )
            )
            fig3.add_hline(y=0, line_color=C_GRID)
            fig3.update_layout(xaxis_title="", yaxis_title="MWh")
            _style(fig3, h=380)
            st.plotly_chart(fig3, use_container_width=True)

    # NG price
    st.subheader(
        "Natural Gas Price",
        help="Henry Hub spot price ($/MMBtu).",
    )
    ng = _D["ng_price"]
    if not ng.empty:
        fig4 = go.Figure(
            go.Scatter(
                x=ng["date"],
                y=ng["ng_price"],
                line=dict(color=C_AMBER, width=2),
                fill="tozeroy",
                fillcolor="rgba(217,119,6,0.06)",
            )
        )
        fig4.update_layout(xaxis_title="", yaxis_title="$/MMBtu")
        _style(fig4, h=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.caption(f"Loaded in {time.time() - t0:.2f}s")


# ===================================================================
# 📄 ABOUT
# ===================================================================
elif page == "📄 About":
    t0 = time.time()
    st.title("📄 About")

    st.markdown(
        """
        # ⚡ Grid Intelligence Platform

        **Open-source power market intelligence for monitoring U.S. electricity system behavior across major balancing authorities**

        This project turns public electricity system data into practical, decision-oriented analytics.  
        The platform is designed to help users understand grid conditions, forecast performance, interregional electricity flows, and renewable energy opportunities without relying on proprietary market intelligence tools.

        ---
        ## What this platform does

        The dashboard analyzes **10 major U.S. balancing authorities (BAs)**:

        - **CISO** — California ISO
        - **ERCO** — ERCOT (Texas)
        - **PJM** — PJM Interconnection
        - **MISO** — Midcontinent ISO
        - **NYIS** — New York ISO
        - **ISNE** — ISO New England
        - **SWPP** — Southwest Power Pool
        - **SOCO** — Southern Company
        - **TVA** — Tennessee Valley Authority
        - **BPAT** — Bonneville Power Administration

        It is organized around five analytical modules:

        1. **Market Overview** — latest demand, forecast accuracy, generation mix, interchange, and gas price context  
        2. **Anomaly Detection** — identifies unusually large demand forecast errors relative to historical patterns  
        3. **Arbitrage Signals** — detects persistent directional interchange flows that may indicate cross-market imbalance  
        4. **Renewable Siting** — scores regions on demand growth, renewable headroom, import dependence, and fossil transition opportunity  
        5. **Compliance Reports** — generates structured balancing-authority-level operational summaries

        ---
        ## Data sources

        This app integrates **five external source datasets**:

        ### 1) EIA Demand & Forecast
        - **Endpoint:** `electricity/rto/region-data`
        - **Frequency:** Hourly
        - **Content:** Actual demand and day-ahead demand forecast by balancing authority

        ### 2) EIA Interchange
        - **Endpoint:** `electricity/rto/interchange-data`
        - **Frequency:** Hourly
        - **Content:** Electricity flows between balancing authorities

        ### 3) EIA Generation by Fuel Type
        - **Endpoint:** `electricity/rto/fuel-type-data`
        - **Frequency:** Hourly
        - **Content:** Electricity generation by fuel category

        ### 4) EIA Natural Gas Price
        - **Endpoint:** `natural-gas/pri/fut`
        - **Frequency:** Daily
        - **Content:** Henry Hub natural gas prices

        ### 5) Open-Meteo Weather
        - **API:** Archive weather API
        - **Frequency:** Daily
        - **Content:** Daily max, min, and average temperature for representative BA locations

        Fuel categories tracked in the generation dataset include:

        - **NG** — Natural Gas  
        - **SUN** — Solar  
        - **WND** — Wind  
        - **NUC** — Nuclear  
        - **COL** — Coal  
        - **WAT** — Hydro  
        - **OIL** — Oil  
        - **OTH** — Other

        ---
        ## Datasets used in the app

        The ETL pipeline writes data into the BigQuery dataset **`eia_data`** in project **`sipa-adv-c-silly-penguin`**.

        ### Raw datasets
        - `hourly_demand`
        - `hourly_interchange`
        - `hourly_fuel_type`
        - `daily_ng_price`
        - `daily_weather`

        ### Derived / summary datasets
        - `daily_demand_summary`
        - `daily_interchange_summary`
        - `daily_fuel_summary`
        - `ba_mape_ranking`

        The Streamlit app currently reads the main raw tables plus the ranking table directly into memory for dashboard use.

        ---
        ## Data pipeline

        The platform uses a **two-stage architecture**:

        **External APIs → ETL pipeline → BigQuery → Streamlit dashboard**

        ### Stage 1: ETL pipeline
        The script `load_to_bigquery.py`:
        - fetches source data from EIA and Open-Meteo
        - writes raw data into BigQuery
        - creates pre-computed summary tables for faster analysis

        The ETL uses a rolling **3-month window** and performs a **full refresh** rather than append logic.

        ### Stage 2: Dashboard loading
        The Streamlit app loads all major datasets once at startup using `st.cache_resource`, then serves all pages from in-memory DataFrames for faster navigation across modules.

        ---
        ## Methodology

        ### Forecast Error Monitoring
        The anomaly module compares actual demand with day-ahead demand forecast, calculates forecast error, and flags balancing authorities whose recent errors exceed their own historical thresholds.

        ### Interchange-Based Arbitrage Signals
        The arbitrage module evaluates BA-to-BA interchange routes using directional strength and consistency, highlighting persistent peak-hour flow patterns that may reflect market imbalance.

        ### Renewable Siting Score
        Renewable investment opportunity is scored using four equal components:
        - demand growth
        - renewable headroom
        - import dependence
        - fossil transition opportunity

        ### Validation
        Source datasets are validated with Pandera schemas covering:
        - demand data
        - interchange data
        - fuel type data
        - natural gas prices
        - weather data
        - merged daily datasets

        ---
        ## Limitations

        - EIA data is public operational data, not full real-time market pricing data
        - The platform does **not** include nodal LMP prices
        - Weather is represented by one reference location per balancing authority
        - Arbitrage signals reflect flow patterns rather than confirmed price spreads
        - The rolling 3-month window is better for short-horizon operational analysis than long-term structural inference

        ---
        ## Tech stack

        - **Frontend:** Streamlit, Plotly
        - **Pipeline:** Python ETL
        - **Data warehouse:** Google BigQuery
        - **Validation:** Pandera
        - **Deployment:** Streamlit Cloud

        ---
        ## Team

        **Xingyi Wang & Wuhao Xia**  
        Columbia SIPA — Advanced Computing for Policy
        """
    )

    st.caption(f"Loaded in {time.time() - t0:.2f}s")
