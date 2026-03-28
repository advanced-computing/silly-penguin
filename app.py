"""
EIA Grid Monitor Dashboard
How Do Power Systems Respond to Uncertainty and Shocks?

Pages:
1. Executive Overview
2. Forecast Uncertainty
3. System Flexibility (Interchange)
4. Fuel Substitution & Price Response
5. Geographic Dashboard
6. Research & Methodology
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import (
    build_geographic_summary,
    calculate_grid_kpis,
    compute_daily_demand_stats,
    compute_daily_interchange,
    compute_error_heatmap_data,
    compute_fuel_share,
    compute_generation_mix,
    compute_hourly_interchange_pattern,
    compute_net_interchange,
    compute_ng_price_vs_generation,
    merge_demand_weather,
    prepare_demand_pivot,
)

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="US Grid Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

HTTP_OK = 200

# ==========================================
# BigQuery Connection
# ==========================================
GCP_PROJECT = "sipa-adv-c-silly-penguin"
BQ_DATASET = "eia_data"


def _get_bq_credentials():
    """Build credentials from Streamlit secrets or fall back to ADC."""
    try:
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
    except (FileNotFoundError, KeyError):
        return None  # falls back to Application Default Credentials


_bq_creds = _get_bq_credentials()

# ==========================================
# BA Constants
# ==========================================
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


# ==========================================
# Data Loading: one-time upfront load, then all pages read from memory
# ==========================================
def _get_bq_client():
    """Create a BigQuery client."""
    from google.cloud import bigquery

    if _bq_creds is not None:
        return bigquery.Client(project=GCP_PROJECT, credentials=_bq_creds)
    return bigquery.Client(project=GCP_PROJECT)


_DS = f"{GCP_PROJECT}.{BQ_DATASET}"


@st.cache_resource(show_spinner="Loading data from BigQuery (one-time)...")
def _load_all_data():
    """Load all tables once at app startup. Stays in memory across reruns."""
    client = _get_bq_client()

    def q(sql):
        return client.query(sql).to_dataframe()

    data = {}

    # Demand (single query, all BAs)
    df = q(
        "SELECT period, respondent, `type-name`, CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_demand`"
    )
    df["period"] = pd.to_datetime(df["period"])
    data["demand"] = df

    # Interchange
    df = q(
        "SELECT period, fromba, toba, CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_interchange`"
    )
    df["period"] = pd.to_datetime(df["period"])
    data["interchange"] = df

    # Fuel type
    df = q(
        "SELECT period, respondent, fueltype, CAST(value AS FLOAT64) AS value "
        f"FROM `{_DS}.hourly_fuel_type`"
    )
    df["period"] = pd.to_datetime(df["period"])
    data["fuel"] = df

    # NG prices
    df = q(f"SELECT date, CAST(ng_price AS FLOAT64) AS ng_price FROM `{_DS}.daily_ng_price`")
    df["date"] = pd.to_datetime(df["date"])
    data["ng_price"] = df

    # Weather
    df = q(
        "SELECT date, ba, "
        "CAST(avg_temp AS FLOAT64) AS avg_temp, "
        "CAST(max_temp AS FLOAT64) AS max_temp, "
        "CAST(min_temp AS FLOAT64) AS min_temp "
        f"FROM `{_DS}.daily_weather`"
    )
    df["date"] = pd.to_datetime(df["date"])
    data["weather"] = df

    # MAPE ranking (pre-aggregated, tiny)
    data["ranking"] = q(f"SELECT * FROM `{_DS}.ba_mape_ranking` ORDER BY mape")

    return data


# Load once — persists across all page navigations and reruns
_data = _load_all_data()


# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("⚡ US Grid Monitor")
st.sidebar.caption("How Do Power Systems Respond to Uncertainty and Shocks?")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Executive Overview",
        "🎯 Forecast Uncertainty",
        "🔄 System Flexibility",
        "⛽ Fuel Substitution",
        "🗺️ Geographic Dashboard",
        "📄 Research & Methodology",
    ],
)

st.sidebar.markdown("---")
days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)

# BA selector for detail pages
selected_ba = st.sidebar.selectbox(
    "Select Balancing Authority",
    MAJOR_BA,
    format_func=lambda x: f"{x} — {BA_NAMES.get(x, x)}",
)


# ==========================================
# PAGE 1: Executive Overview
# ==========================================
if page == "📊 Executive Overview":
    start_time = time.time()
    st.title("📊 US Grid Monitor")
    st.markdown(
        """
        ### How Do Power Systems Respond to Uncertainty and Shocks?

        This dashboard analyzes US electricity grid behavior across **10 major
        balancing authorities** using hourly data from the Energy Information
        Administration (EIA). Explore three dimensions of grid resilience:
        """
    )

    st.markdown("---")

    dim_cols = st.columns(3)
    with dim_cols[0]:
        st.markdown(
            """
            #### 🎯 Forecast Uncertainty
            How accurate are day-ahead demand forecasts?
            When and where do the largest errors occur?

            → Select **Forecast Uncertainty** in the sidebar
            """
        )
    with dim_cols[1]:
        st.markdown(
            """
            #### 🔄 System Flexibility
            Do regions rely more on power imports during
            high-demand periods? What are the intra-day patterns?

            → Select **System Flexibility** in the sidebar
            """
        )
    with dim_cols[2]:
        st.markdown(
            """
            #### ⛽ Fuel Substitution
            How does the generation mix shift when natural gas
            prices change? Which fuels serve as substitutes?

            → Select **Fuel Substitution** in the sidebar
            """
        )

    st.markdown("---")

    st.subheader("Balancing Authorities Covered")
    ba_table = pd.DataFrame([{"Code": k, "Name": v} for k, v in BA_NAMES.items()])
    st.dataframe(ba_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        """
        **Data Sources:** EIA Hourly Demand & Forecast · EIA Interchange ·
        EIA Generation by Fuel Type · Henry Hub Natural Gas Prices · Open-Meteo Weather

        **Team:** Xingyi Wang & Wuhao Xia — SIPA, Columbia University
        """
    )

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")


# ==========================================
# PAGE 2: Forecast Uncertainty
# ==========================================
elif page == "🎯 Forecast Uncertainty":
    start_time = time.time()
    st.title("🎯 Forecast Uncertainty Analysis")
    st.markdown(
        f"""
        Analyzing demand forecast errors for **{BA_NAMES.get(selected_ba, selected_ba)}**.
        Use the sidebar to switch between balancing authorities and adjust the time window.
        """
    )

    demand_df = _data["demand"]
    weather_df = _data["weather"]

    if demand_df.empty:
        st.warning("No demand data available.")
    else:
        pivot = prepare_demand_pivot(demand_df, selected_ba)

        if pivot.empty:
            st.warning(f"No demand data found for {selected_ba}.")
        else:
            # KPIs
            actual, forecast, delta = calculate_grid_kpis(pivot)
            if actual is not None:
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Latest Demand", f"{actual:,.0f} MWh")
                kpi_cols[1].metric("Latest Forecast", f"{forecast:,.0f} MWh")
                kpi_cols[2].metric("Forecast Error", f"{delta:,.0f} MWh")
                if "APE" in pivot.columns:
                    avg_mape = pivot["APE"].mean()
                    kpi_cols[3].metric("Avg MAPE", f"{avg_mape:.2f}%")

            st.markdown("---")

            # Forecast vs Actual time series
            st.subheader(f"Demand vs Forecast — Last {days_to_show} Days")
            df_display = pivot.head(days_to_show * 24).sort_index()

            if {"Demand", "Day-ahead demand forecast"}.issubset(df_display.columns):
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_display.index,
                        y=df_display["Demand"],
                        name="Actual Demand",
                        line={"color": "#2196F3", "width": 2},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_display.index,
                        y=df_display["Day-ahead demand forecast"],
                        name="Day-ahead Forecast",
                        line={"color": "#FF9800", "width": 2, "dash": "dash"},
                    )
                )
                fig.update_layout(
                    height=450,
                    xaxis_title="Time",
                    yaxis_title="Demand (MWh)",
                    hovermode="x unified",
                    legend={"orientation": "h", "y": -0.15},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Error trend with MA
            if "Forecast Error" in df_display.columns:
                st.subheader("Forecast Error Trend")
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(
                        x=df_display.index,
                        y=df_display["Forecast Error"],
                        name="Hourly Error",
                        line={"color": "#E91E63", "width": 1},
                        opacity=0.6,
                    )
                )
                if "Error 24h MA" in df_display.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=df_display.index,
                            y=df_display["Error 24h MA"],
                            name="24h Moving Average",
                            line={"color": "#4CAF50", "width": 2.5},
                        )
                    )
                fig2.add_hline(y=0, line_dash="dot", line_color="gray")
                fig2.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Forecast Error (MWh)",
                    hovermode="x unified",
                    legend={"orientation": "h", "y": -0.15},
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Error heatmap
            st.subheader("Forecast Error Heatmap (Hour × Date)")
            heatmap_data = compute_error_heatmap_data(demand_df, selected_ba)
            if not heatmap_data.empty:
                fig3 = px.imshow(
                    heatmap_data.T,
                    labels={"x": "Date", "y": "Hour of Day", "color": "Error (MWh)"},
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    aspect="auto",
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    "Red = demand exceeded forecast; Blue = forecast exceeded demand. "
                    "Patterns reveal systematic time-of-day biases."
                )

            # BA MAPE ranking
            st.subheader("Cross-BA Forecast Accuracy Comparison")
            ranking = _data["ranking"]
            if not ranking.empty:
                ranking["ba_name"] = ranking["ba"].map(BA_NAMES)
                fig4 = px.bar(
                    ranking,
                    x="mape",
                    y="ba_name",
                    orientation="h",
                    title="MAPE Ranking Across Balancing Authorities",
                    labels={"mape": "MAPE (%)", "ba_name": ""},
                    color="mape",
                    color_continuous_scale="RdYlGn_r",
                )
                fig4.update_layout(
                    height=400,
                    yaxis={"categoryorder": "total ascending"},
                    showlegend=False,
                )
                st.plotly_chart(fig4, use_container_width=True)

            # Weather impact
            st.subheader("Temperature Sensitivity")
            daily_stats = compute_daily_demand_stats(demand_df, selected_ba)
            if not daily_stats.empty and not weather_df.empty:
                merged = merge_demand_weather(daily_stats, weather_df, selected_ba)
                if not merged.empty:
                    weather_cols = st.columns(2)
                    with weather_cols[0]:
                        fig5 = px.scatter(
                            merged,
                            x="avg_temp",
                            y="avg_demand_mwh",
                            title="Temperature vs Demand",
                            labels={
                                "avg_temp": "Avg Temperature (°C)",
                                "avg_demand_mwh": "Avg Demand (MWh)",
                            },
                            trendline="ols",
                            opacity=0.7,
                        )
                        fig5.update_layout(height=400)
                        st.plotly_chart(fig5, use_container_width=True)

                    with weather_cols[1]:
                        fig6 = px.scatter(
                            merged,
                            x="avg_temp",
                            y="avg_error_mwh",
                            title="Temperature vs Forecast Error",
                            labels={
                                "avg_temp": "Avg Temperature (°C)",
                                "avg_error_mwh": "Avg Error (MWh)",
                            },
                            trendline="ols",
                            opacity=0.7,
                        )
                        fig6.update_layout(height=400)
                        st.plotly_chart(fig6, use_container_width=True)

            with st.expander("📊 View Raw Hourly Data"):
                st.dataframe(df_display, use_container_width=True)

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")


# ==========================================
# PAGE 3: System Flexibility (Interchange)
# ==========================================
elif page == "🔄 System Flexibility":
    start_time = time.time()
    st.title("🔄 System Flexibility — Interregional Power Trade")
    st.markdown(
        f"""
        Analyzing how **{BA_NAMES.get(selected_ba, selected_ba)}** relies on
        interregional power interchange as a flexibility mechanism.
        Positive values = net import; Negative = net export.
        """
    )

    interchange_df = _data["interchange"]
    demand_df = _data["demand"]

    if interchange_df.empty or "fromba" not in interchange_df.columns:
        st.warning("No interchange data available.")
    else:
        # Net interchange time series
        st.subheader(f"{selected_ba} Net Interchange — Last {days_to_show} Days")
        net_int = compute_net_interchange(interchange_df, selected_ba)
        if not net_int.empty:
            recent = net_int.tail(days_to_show * 24)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=recent["period"],
                    y=recent["net_interchange_mwh"],
                    fill="tozeroy",
                    name="Net Interchange",
                    line={"color": "#00BCD4", "width": 1.5},
                    fillcolor="rgba(0, 188, 212, 0.2)",
                )
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(
                height=450,
                xaxis_title="Time",
                yaxis_title="Net Interchange (MWh)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Above zero = net import from other regions; below zero = net export.")

        # Daily interchange trend
        st.subheader("Daily Interchange Summary")
        daily_int = compute_daily_interchange(interchange_df, selected_ba)
        if not daily_int.empty:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=daily_int["date"],
                    y=daily_int["avg_interchange"],
                    name="Daily Average",
                    line={"color": "#00BCD4", "width": 2},
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=daily_int["date"],
                    y=daily_int["max_interchange"],
                    name="Daily Max",
                    line={"color": "#FF5722", "width": 1, "dash": "dot"},
                    opacity=0.5,
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=daily_int["date"],
                    y=daily_int["min_interchange"],
                    name="Daily Min",
                    line={"color": "#4CAF50", "width": 1, "dash": "dot"},
                    opacity=0.5,
                )
            )
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
            fig2.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Interchange (MWh)",
                hovermode="x unified",
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(fig2, use_container_width=True)

        int_detail_cols = st.columns(2)

        # Hourly pattern
        with int_detail_cols[0]:
            st.subheader("Hourly Interchange Pattern")
            hourly_pat = compute_hourly_interchange_pattern(interchange_df, selected_ba)
            if not hourly_pat.empty:
                fig3 = px.bar(
                    hourly_pat,
                    x="hour",
                    y="avg_interchange_mwh",
                    title="Average Interchange by Hour of Day",
                    labels={
                        "hour": "Hour",
                        "avg_interchange_mwh": "Avg MWh",
                    },
                    color="avg_interchange_mwh",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                )
                fig3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Reveals which hours the region imports vs exports power.")

        # Demand vs Interchange scatter
        with int_detail_cols[1]:
            st.subheader("Demand vs Net Interchange")
            if not demand_df.empty:
                ba_demand = demand_df[
                    (demand_df["respondent"] == selected_ba) & (demand_df["type-name"] == "Demand")
                ].copy()
                if not ba_demand.empty and not net_int.empty:
                    scatter_data = ba_demand.merge(net_int, on="period", how="inner")
                    if not scatter_data.empty:
                        fig4 = px.scatter(
                            scatter_data,
                            x="value",
                            y="net_interchange_mwh",
                            title="Demand Level vs Net Interchange",
                            labels={
                                "value": "Demand (MWh)",
                                "net_interchange_mwh": "Net Interchange (MWh)",
                            },
                            trendline="ols",
                            opacity=0.4,
                        )
                        fig4.update_layout(height=400)
                        st.plotly_chart(fig4, use_container_width=True)
                        st.caption(
                            "Tests whether higher demand correlates with "
                            "greater reliance on imported power."
                        )

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")


# ==========================================
# PAGE 4: Fuel Substitution & Price Response
# ==========================================
elif page == "⛽ Fuel Substitution":
    start_time = time.time()
    st.title("⛽ Fuel Substitution & Price Response")
    st.markdown(
        f"""
        Analyzing the generation mix for **{BA_NAMES.get(selected_ba, selected_ba)}**
        and how fuel composition responds to energy price shocks.
        """
    )

    fuel_df = _data["fuel"]
    ng_price_df = _data["ng_price"]

    if fuel_df.empty or "fueltype" not in fuel_df.columns:
        st.warning("No fuel type data available.")
    else:
        # Generation mix stacked area
        st.subheader("Daily Generation Mix")
        mix = compute_generation_mix(fuel_df, selected_ba)
        if not mix.empty:
            mix_display = mix.copy()
            mix_display["fuel_label"] = mix_display["fueltype"].map(FUEL_LABELS)

            fig = px.area(
                mix_display,
                x="date",
                y="avg_generation_mwh",
                color="fuel_label",
                title=f"{selected_ba} Generation by Fuel Type",
                labels={
                    "avg_generation_mwh": "Avg Generation (MWh)",
                    "date": "Date",
                    "fuel_label": "Fuel Type",
                },
                color_discrete_map={
                    "Natural Gas": "#FF9800",
                    "Solar": "#FFC107",
                    "Wind": "#03A9F4",
                    "Nuclear": "#9C27B0",
                    "Coal": "#795548",
                    "Hydro": "#2196F3",
                    "Oil": "#F44336",
                    "Other": "#9E9E9E",
                },
            )
            fig.update_layout(
                height=500,
                hovermode="x unified",
                legend={"orientation": "h", "y": -0.2},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Fuel share trends
        st.subheader("Fuel Share Trends (%)")
        share = compute_fuel_share(fuel_df, selected_ba)
        if not share.empty:
            share_display = share.copy()
            share_display["fuel_label"] = share_display["fueltype"].map(FUEL_LABELS)

            fig2 = px.line(
                share_display,
                x="date",
                y="share_pct",
                color="fuel_label",
                title=f"{selected_ba} Fuel Share Over Time",
                labels={
                    "share_pct": "Share (%)",
                    "date": "Date",
                    "fuel_label": "Fuel Type",
                },
            )
            fig2.update_layout(
                height=450,
                hovermode="x unified",
                legend={"orientation": "h", "y": -0.2},
            )
            st.plotly_chart(fig2, use_container_width=True)

        # NG price vs generation
        st.subheader("Natural Gas Price vs Generation Response")
        if not ng_price_df.empty:
            price_gen = compute_ng_price_vs_generation(fuel_df, ng_price_df, selected_ba)
            if not price_gen.empty:
                # Dual-axis: NG price + NG generation share
                ng_data = price_gen[price_gen["fueltype"] == "NG"]
                if not ng_data.empty:
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig3.add_trace(
                        go.Scatter(
                            x=ng_data["date"],
                            y=ng_data["ng_price"],
                            name="Henry Hub Price ($/MMBtu)",
                            line={"color": "#F44336", "width": 2},
                        ),
                        secondary_y=False,
                    )
                    fig3.add_trace(
                        go.Scatter(
                            x=ng_data["date"],
                            y=ng_data["share_pct"],
                            name="NG Generation Share (%)",
                            line={"color": "#FF9800", "width": 2},
                        ),
                        secondary_y=True,
                    )
                    fig3.update_layout(
                        title="Natural Gas Price vs Generation Share",
                        height=450,
                        hovermode="x unified",
                        legend={"orientation": "h", "y": -0.15},
                    )
                    fig3.update_yaxes(title_text="Price ($/MMBtu)", secondary_y=False)
                    fig3.update_yaxes(title_text="NG Share (%)", secondary_y=True)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.caption(
                        "When natural gas prices spike, do other fuels "
                        "increase their share? This dual-axis chart reveals "
                        "the price-generation relationship."
                    )

                # Price elasticity scatter by fuel
                st.subheader("Price Sensitivity by Fuel Type")
                fuel_types_to_plot = ["NG", "COL", "SUN", "WND", "NUC", "WAT", "OIL"]
                scatter_frames = []
                for ft in fuel_types_to_plot:
                    ft_data = price_gen[price_gen["fueltype"] == ft].copy()
                    if not ft_data.empty:
                        ft_data["fuel_label"] = ft_data["fueltype"].map(FUEL_LABELS)
                        scatter_frames.append(ft_data)

                if scatter_frames:
                    all_scatter = pd.concat(scatter_frames, ignore_index=True)
                    fig4 = px.scatter(
                        all_scatter,
                        x="ng_price",
                        y="share_pct",
                        color="fuel_label",
                        title="NG Price vs Fuel Generation Share",
                        labels={
                            "ng_price": "Henry Hub Price ($/MMBtu)",
                            "share_pct": "Generation Share (%)",
                            "fuel_label": "Fuel Type",
                        },
                        trendline="ols",
                        opacity=0.5,
                    )
                    fig4.update_layout(
                        height=500,
                        legend={"orientation": "h", "y": -0.2},
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption(
                        "Each point is one day. Trend lines show whether a fuel type's "
                        "share increases or decreases as gas prices rise — revealing "
                        "fuel substitution dynamics."
                    )
        else:
            st.info("Natural gas price data not available for price-response analysis.")

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")


# ==========================================
# PAGE 5: Geographic Dashboard
# ==========================================
elif page == "🗺️ Geographic Dashboard":
    start_time = time.time()
    st.title("🗺️ Geographic Dashboard")
    st.markdown(
        """
        A map-based view of power system performance across major US balancing authorities.
        Select a metric to visualize regional differences.
        """
    )

    demand_df = _data["demand"]
    interchange_df = _data["interchange"]
    fuel_df = _data["fuel"]

    geo_summary = build_geographic_summary(demand_df, interchange_df, fuel_df)

    if geo_summary.empty:
        st.warning("Geographic data not available.")
    else:
        metric_option = st.selectbox(
            "Select map metric",
            ["mape", "avg_demand", "avg_net_interchange", "renewable_share"],
            format_func=lambda x: {
                "mape": "Forecast MAPE (%)",
                "avg_demand": "Average Demand (MWh)",
                "avg_net_interchange": "Avg Net Interchange (MWh)",
                "renewable_share": "Renewable Share (%)",
            }.get(x, x),
        )

        # Filter to rows with the chosen metric
        plot_df = geo_summary.dropna(subset=[metric_option])

        if plot_df.empty:
            st.warning(f"No data for metric: {metric_option}")
        else:
            fig = px.scatter_geo(
                plot_df,
                lat="lat",
                lon="lon",
                size=plot_df[metric_option].abs(),
                color=metric_option,
                hover_name="name",
                hover_data={
                    "ba": True,
                    "mape": ":.2f",
                    "avg_demand": ":.0f",
                    "renewable_share": ":.1f",
                    "lat": False,
                    "lon": False,
                },
                title=f"US Balancing Authorities — {metric_option}",
                color_continuous_scale="Viridis",
            )
            fig.update_geos(
                scope="usa",
                showlakes=True,
                lakecolor="rgb(200, 220, 255)",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Regional Summary Table")
        display_cols = [
            c
            for c in [
                "ba",
                "name",
                "mape",
                "avg_demand",
                "avg_net_interchange",
                "renewable_share",
            ]
            if c in geo_summary.columns
        ]
        styled = geo_summary[display_cols].copy()
        st.dataframe(
            styled.style.format(
                {
                    "mape": "{:.2f}%",
                    "avg_demand": "{:,.0f}",
                    "avg_net_interchange": "{:,.0f}",
                    "renewable_share": "{:.1f}%",
                },
                na_rep="—",
            ),
            use_container_width=True,
        )

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")


# ==========================================
# PAGE 6: Research & Methodology
# ==========================================
elif page == "📄 Research & Methodology":
    start_time = time.time()
    st.title("📄 Research & Methodology")
    st.caption("Course Project | SIPA Advanced Computing for Policy")

    st.header("Research Question")
    st.markdown(
        """
        **How do power systems respond to uncertainty and shocks?**

        We investigate this through three analytical dimensions:
        """
    )

    dim_cols = st.columns(3)
    with dim_cols[0]:
        st.markdown(
            """
            #### 🎯 Forecast Uncertainty
            How accurate are day-ahead demand forecasts?
            When and where do the largest errors occur?
            How does temperature affect prediction accuracy?
            """
        )
    with dim_cols[1]:
        st.markdown(
            """
            #### 🔄 System Flexibility
            Do balancing authorities rely more on interregional
            power imports during high-demand periods?
            What are the intra-day patterns of power trade?
            """
        )
    with dim_cols[2]:
        st.markdown(
            """
            #### ⛽ Fuel Substitution
            How does the generation mix shift when natural gas
            prices change? Which fuels serve as substitutes?
            Do regions differ in their fuel switching behavior?
            """
        )

    st.markdown("---")

    st.header("Data Sources")
    st.markdown(
        """
        | Source | Endpoint | Frequency | Coverage |
        |--------|----------|-----------|----------|
        | EIA Demand & Forecast | `electricity/rto/region-data` | Hourly | 10 major BAs |
        | EIA Interchange | `electricity/rto/interchange-data` | Hourly | BA-to-BA flows |
        | EIA Generation by Fuel | `electricity/rto/fuel-type-data` | Hourly | By fuel type |
        | EIA Natural Gas Price | `natural-gas/pri/fut` | Daily | Henry Hub spot |
        | Open-Meteo Weather | Archive API | Daily | Representative cities |
        """
    )

    st.header("Balancing Authorities Covered")
    ba_table = pd.DataFrame([{"Code": k, "Name": v} for k, v in BA_NAMES.items()])
    st.dataframe(ba_table, use_container_width=True, hide_index=True)

    st.header("Methodology")
    st.markdown(
        """
        **Forecast Uncertainty**: We compute Mean Absolute Percentage Error (MAPE),
        Mean Absolute Error (MAE), and hourly error patterns across all balancing
        authorities. Temperature sensitivity is assessed through scatter analysis
        with OLS trend lines.

        **System Flexibility**: Net interchange is computed by summing all flows
        from a BA to its neighbors. We analyze correlation between demand levels
        and net imports, and identify hour-of-day patterns.

        **Fuel Substitution**: We track the daily generation share of each fuel type
        and overlay Henry Hub natural gas spot prices. Price-generation elasticity
        is estimated via cross-sectional scatter analysis across fuel types.
        """
    )

    st.header("Limitations")
    st.markdown(
        """
        - Not all BAs report consistently across all data types
        - Weather data uses single representative cities per BA region
        - Henry Hub price is a national proxy; regional gas prices may differ
        - Hourly data may contain gaps or reporting delays from EIA
        - Fuel substitution analysis assumes contemporaneous response;
          actual switching may involve lags
        """
    )

    st.header("Team")
    st.markdown("**Xingyi Wang & Wuhao Xia** — SIPA, Columbia University")

    elapsed = time.time() - start_time
    st.caption(f"Page loaded in {elapsed:.2f} seconds")
