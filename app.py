import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

from data_processing import calculate_grid_kpis, merge_daily_demand_weather

st.set_page_config(page_title="EIA Grid Monitor", page_icon="⚡", layout="wide")

# ==========================================
# Load API Key
# ==========================================
load_dotenv()
api_key = os.getenv("EIA_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["EIA_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("API Key not found. Please set it in .env (Local) or Streamlit Secrets (Cloud).")
        st.stop()


# ==========================================
# Data Fetching Functions
# ==========================================
@st.cache_data(ttl=3600)
def get_eia_data(api_key):
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "CISO",
        "facets[type][]": ["D", "DF"],  # D=Actual Demand, DF=Forecast
        "start": "2025-01-01T00",
        "end": "2026-02-01T00",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 200:
        data = response.json()["response"]["data"]
        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["period"] = pd.to_datetime(df["period"])
        return df
    st.error(f"EIA API Error: {response.status_code}")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 34.05,
        "longitude": -118.24,
        "start_date": "2025-01-01",
        "end_date": "2026-02-01",
        "daily": ["temperature_2m_max", "temperature_2m_min", "weathercode"],
        "timezone": "America/Los_Angeles",
    }
    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 200:
        data = response.json()
        daily = data["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "max_temp": pd.to_numeric(daily["temperature_2m_max"], errors="coerce"),
                "min_temp": pd.to_numeric(daily["temperature_2m_min"], errors="coerce"),
            }
        )
        df["avg_temp"] = (df["max_temp"] + df["min_temp"]) / 2
        return df
    st.error("Weather API Error")
    return pd.DataFrame()


# ==========================================
# Helper Functions
# ==========================================
def prepare_hourly_pivot(df):
    """Convert long EIA data into pivoted hourly format."""
    if df.empty:
        return pd.DataFrame()

    df_pivot = df.pivot_table(index="period", columns="type-name", values="value")
    df_pivot = df_pivot.sort_index(ascending=False)

    if {"Demand", "Day-ahead demand forecast"}.issubset(df_pivot.columns):
        df_pivot["Forecast Error"] = df_pivot["Demand"] - df_pivot["Day-ahead demand forecast"]
        df_pivot["Absolute Error"] = df_pivot["Forecast Error"].abs()
        df_pivot["APE"] = (df_pivot["Absolute Error"] / df_pivot["Demand"].replace(0, pd.NA)) * 100
        df_pivot["Error 24h MA"] = df_pivot["Forecast Error"].rolling(24).mean()

    return df_pivot


def prepare_daily_error_weather(eia_df, weather_df):
    """Create daily dataset with demand, forecast, error, and weather."""
    if eia_df.empty or weather_df.empty:
        return pd.DataFrame()

    pivot = eia_df.pivot_table(index="period", columns="type-name", values="value")
    pivot = pivot.sort_index()

    if not {"Demand", "Day-ahead demand forecast"}.issubset(pivot.columns):
        return pd.DataFrame()

    pivot["error"] = pivot["Demand"] - pivot["Day-ahead demand forecast"]
    pivot["abs_error"] = pivot["error"].abs()
    pivot["ape"] = (pivot["abs_error"] / pivot["Demand"].replace(0, pd.NA)) * 100
    pivot["date"] = pd.to_datetime(pivot.index.date)

    daily = (
        pivot.groupby("date")
        .agg(
            avg_demand_mwh=("Demand", "mean"),
            avg_forecast_mwh=("Day-ahead demand forecast", "mean"),
            avg_error_mwh=("error", "mean"),
            mape=("ape", "mean"),
        )
        .reset_index()
    )

    merged = daily.merge(weather_df, on="date", how="inner")
    return merged


def label_weather_regime(df):
    """Classify days into cold / mild / hot using quantiles."""
    if df.empty or "avg_temp" not in df.columns:
        return df

    df = df.copy()
    low_q = df["avg_temp"].quantile(0.1)
    high_q = df["avg_temp"].quantile(0.9)

    def classify(temp):
        if temp <= low_q:
            return "Cold Extreme"
        if temp >= high_q:
            return "Hot Extreme"
        return "Mild / Normal"

    df["weather_regime"] = df["avg_temp"].apply(classify)
    return df


def build_region_summary(df_pivot):
    """Create summary dataframe for map visualization."""
    if df_pivot.empty:
        return pd.DataFrame()

    latest_row = df_pivot.iloc[0]

    summary = pd.DataFrame(
        {
            "region": ["CISO"],
            "region_name": ["California ISO"],
            "lat": [36.7783],
            "lon": [-119.4179],
            "actual_demand": [latest_row.get("Demand", pd.NA)],
            "forecast_demand": [latest_row.get("Day-ahead demand forecast", pd.NA)],
            "forecast_error": [latest_row.get("Forecast Error", pd.NA)],
            "absolute_error": [latest_row.get("Absolute Error", pd.NA)],
        }
    )
    return summary


# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📊 Grid Overview",
        "⚡ Grid Monitor",
        "🌡️ Weather Impact Analysis",
        "🗺️ Map Dashboard",
        "📄 Project Proposal",
    ],
)

# Shared filter
st.sidebar.markdown("---")
days_to_show = st.sidebar.slider("Days to visualize", 1, 30, 7)

# ==========================================
# Load Data Once
# ==========================================
with st.spinner("Loading electricity and weather data..."):
    eia_df = get_eia_data(api_key)
    weather_df = get_weather_data()
    df_pivot = prepare_hourly_pivot(eia_df)
    merged_df = prepare_daily_error_weather(eia_df, weather_df)
    merged_df = label_weather_regime(merged_df)

# ==========================================
# PAGE 1: Grid Overview
# ==========================================
if page == "📊 Grid Overview":
    st.title("📊 US Electricity Grid Dashboard")
    st.subheader("CISO Demand Monitoring and Forecast Accuracy")
    st.markdown(
        """
        This dashboard monitors **electricity demand**, **day-ahead forecast accuracy**,
        and **weather impacts** using EIA electricity data and Open-Meteo weather data.
        """
    )

    if not df_pivot.empty:
        last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Actual Demand", f"{last_actual:,.0f} MWh")
        col2.metric("Latest Forecast Demand", f"{last_forecast:,.0f} MWh")
        col3.metric("Latest Forecast Error", f"{delta:,.0f} MWh", delta_color="inverse")

        st.markdown("---")
        st.subheader("Recent Demand Trend")

        df_recent = df_pivot.head(days_to_show * 24).sort_index()

        fig, ax = plt.subplots(figsize=(11, 5))
        df_recent[["Demand", "Day-ahead demand forecast"]].plot(ax=ax, linewidth=2)
        ax.set_title("Actual vs Forecast Electricity Demand")
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand (MWh)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(["Actual Demand", "Day-ahead Forecast"])
        st.pyplot(fig)

        st.caption(
            "This chart provides a high-level overview of recent electricity demand and "
            "day-ahead forecast performance in CISO."
        )
    else:
        st.warning("No data available. Please check API access or connection.")


# ==========================================
# PAGE 2: Grid Monitor
# ==========================================
elif page == "⚡ Grid Monitor":
    st.title("⚡ California (CISO) Grid Monitor")
    st.subheader("Xingyi Wang, Wuhao Xia")
    st.markdown(
        "This page focuses on **actual demand**, **day-ahead forecast**, and "
        "**forecast error patterns** in the California ISO region."
    )

    if not df_pivot.empty:
        df_display = df_pivot.head(days_to_show * 24).copy()
        df_display = df_display.sort_index()

        last_actual, last_forecast, delta = calculate_grid_kpis(df_pivot)

        if last_actual is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Actual Demand", f"{last_actual:,.0f} MWh")
            col2.metric("Latest Forecast", f"{last_forecast:,.0f} MWh")
            col3.metric("Forecast Error", f"{delta:,.0f} MWh", delta_color="inverse")
        else:
            st.warning("Data incomplete for KPI calculation.")

        st.subheader(f"Demand vs Forecast (Last {days_to_show} Days)")
        fig1, ax1 = plt.subplots(figsize=(11, 5))
        df_display[["Demand", "Day-ahead demand forecast"]].plot(ax=ax1, linewidth=2)
        ax1.set_title("Actual Demand vs Day-ahead Forecast")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Demand (MWh)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(["Actual Demand", "Day-ahead Forecast"])
        st.pyplot(fig1)

        st.caption("The closer the two lines are, the more accurate the day-ahead demand forecast.")

        st.subheader("Forecast Error Over Time")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        ax2.plot(df_display.index, df_display["Forecast Error"], label="Hourly Forecast Error")
        if "Error 24h MA" in df_display.columns:
            ax2.plot(df_display.index, df_display["Error 24h MA"], label="24-hour Moving Average")
        ax2.set_title("Forecast Error Trend")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Forecast Error (MWh)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        st.caption(
            "Positive values indicate demand exceeded the day-ahead forecast. "
            "The moving average helps reveal persistent patterns rather than hourly noise."
        )

        with st.expander("See Raw Hourly Data"):
            st.dataframe(df_display)
    else:
        st.warning("No data available. Please check API Key or connection.")


# ==========================================
# PAGE 3: Weather Impact Analysis
# ==========================================
elif page == "🌡️ Weather Impact Analysis":
    st.title("🌡️ Weather Impact Analysis")
    st.markdown(
        """
        This page combines **EIA electricity data** with **Open-Meteo weather data**
        to explore how temperature affects electricity demand and forecast accuracy.
        """
    )

    if not merged_df.empty:
        st.subheader("Daily Electricity Demand and Temperature")

        fig3, ax3 = plt.subplots(figsize=(11, 5))
        ax3.plot(merged_df["date"], merged_df["avg_demand_mwh"], label="Daily Avg Demand")
        ax3.set_title("Daily Electricity Demand Trend")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Daily Avg Demand (MWh)")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend()
        st.pyplot(fig3)

        st.caption(
            "This chart shows how average daily electricity demand changes over the study period."
        )

        st.subheader("Temperature Trend")
        fig4, ax4 = plt.subplots(figsize=(11, 4))
        ax4.plot(merged_df["date"], merged_df["avg_temp"], label="Average Temperature")
        ax4.set_title("Daily Average Temperature")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Temperature (°C)")
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend()
        st.pyplot(fig4)

        st.subheader("Temperature vs Electricity Demand")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.scatter(merged_df["avg_temp"], merged_df["avg_demand_mwh"], alpha=0.7)
        ax5.set_title("Temperature Sensitivity of Electricity Demand")
        ax5.set_xlabel("Average Temperature (°C)")
        ax5.set_ylabel("Daily Avg Demand (MWh)")
        ax5.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig5)

        st.caption(
            "This scatter plot shows whether electricity demand tends to increase or decrease "
            "as daily temperature changes."
        )

        st.subheader("Temperature vs Forecast Error")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.scatter(merged_df["avg_temp"], merged_df["avg_error_mwh"], alpha=0.7)
        ax6.set_title("Temperature and Daily Forecast Error")
        ax6.set_xlabel("Average Temperature (°C)")
        ax6.set_ylabel("Average Forecast Error (MWh)")
        ax6.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig6)

        st.caption(
            "This chart helps assess whether hotter or colder days are associated "
            "with larger forecast deviations."
        )

        st.subheader("Daily MAPE Trend")
        fig7, ax7 = plt.subplots(figsize=(11, 5))
        ax7.plot(merged_df["date"], merged_df["mape"], label="Daily MAPE")
        ax7.set_title("Daily Forecast Accuracy (MAPE)")
        ax7.set_xlabel("Date")
        ax7.set_ylabel("MAPE (%)")
        ax7.grid(True, linestyle="--", alpha=0.4)
        ax7.legend()
        st.pyplot(fig7)

        st.caption(
            "MAPE (Mean Absolute Percentage Error) summarizes forecast accuracy. "
            "Higher values indicate worse forecasting performance."
        )

        st.subheader("Extreme vs Mild Weather Comparison")
        regime_summary = (
            merged_df.groupby("weather_regime")[["avg_demand_mwh", "mape", "avg_error_mwh"]]
            .mean()
            .reset_index()
        )
        st.dataframe(regime_summary)

        st.caption(
            "This table compares demand and forecast performance across hot, cold, and mild weather conditions."
        )

        with st.expander("See Daily Weather-Merged Data"):
            st.dataframe(merged_df)

    else:
        st.error("Failed to load merged electricity-weather data for analysis.")


# ==========================================
# PAGE 4: Map Dashboard
# ==========================================
elif page == "🗺️ Map Dashboard":
    st.title("🗺️ Map Dashboard")
    st.markdown(
        """
        This page provides a geographic view of electricity demand forecasting stress.
        For now, the map includes **CISO** as a proof-of-concept region.
        It can later be extended to multiple balancing authorities.
        """
    )

    if not df_pivot.empty:
        region_df = build_region_summary(df_pivot)

        metric_option = st.selectbox(
            "Select map metric",
            ["absolute_error", "forecast_error", "actual_demand", "forecast_demand"],
        )

        fig_map = px.scatter_geo(
            region_df,
            lat="lat",
            lon="lon",
            size=metric_option,
            hover_name="region_name",
            hover_data={
                "actual_demand": True,
                "forecast_demand": True,
                "forecast_error": True,
                "absolute_error": True,
                "lat": False,
                "lon": False,
            },
            title="Balancing Authority Stress Map",
        )
        fig_map.update_geos(scope="usa")
        st.plotly_chart(fig_map, use_container_width=True)

        st.caption(
            "The size of each marker reflects the selected metric. "
            "This page is designed to scale into a multi-region map dashboard."
        )

        st.subheader("Region Summary")
        st.dataframe(region_df)
    else:
        st.warning("No regional summary data available for mapping.")


# ==========================================
# PAGE 5: Project Proposal
# ==========================================
elif page == "📄 Project Proposal":
    st.markdown("---")
    st.caption("Course Project | Advanced Computing for Policy")
    st.title("Project Proposal")
    st.subheader("Electricity Demand Forecasting and Grid Stress Analysis")

    st.header("0. Proposal Updates After Initial Implementation")
    st.write(
        """
        After implementing the first version of the Streamlit dashboard,
        we refined the project focus.

        Originally the proposal included several broad research questions.
        After exploring the data, we narrowed the scope to focus on:

        1. Forecast error in electricity demand prediction  
        2. The relationship between grid stress and interregional electricity interchange

        We also decided to add a **map-based visualization**
        to allow users to explore regional differences
        in prediction accuracy and grid reliance.
        """
    )

    st.header("1. Dataset")
    st.write(
        """
        **Dataset Name:** Balancing Authority Areas Hourly Operating Data

        **Source:** U.S. Energy Information Administration (EIA)

        This dataset provides hourly electricity system data across more than 50
        U.S. balancing authorities, including:

        - Actual electricity demand
        - Day-ahead demand forecasts
        - Net electricity generation
        - Power interchange between regions

        The data is updated hourly and is accessible through the EIA API.
        """
    )

    st.header("2. Research Questions")
    st.write(
        """
        Our project investigates electricity demand forecasting and grid stress.

        **Main Questions:**

        1. How does the forecast error between day-ahead demand forecasts and
           actual electricity demand change during extreme temperature periods?

        2. Do balancing authorities rely more heavily on interregional
           electricity interchange during periods of high demand or
           large prediction errors?

        3. Are there systematic differences in prediction accuracy
           across balancing authorities?
        """
    )

    st.header("3. Target Visualizations")
    st.write(
        """
        The project will produce several visualizations:

        - Time-series line chart comparing **forecasted vs actual demand**
        - Temperature vs electricity demand analysis
        - Forecast error trends over time
        - Scatter plots of temperature vs demand
        - Scatter plots of temperature vs forecast error
        - Map visualization of balancing authorities
          showing prediction error and interchange reliance
        """
    )

    st.header("4. Known Unknowns")
    st.write(
        """
        Some uncertainties remain in the dataset:

        - Not all balancing authorities may report generation breakdown
          consistently across the full time period.

        - Electricity demand can be influenced by factors not captured in the data,
          such as local holidays, outages, or policy changes.

        - Weather data is approximated using representative cities
          rather than full regional weather systems.
        """
    )

    st.header("5. Anticipated Challenges")
    st.write(
        """
        Several technical challenges are expected:

        - Aligning timestamps across multiple regions and datasets
        - Managing large volumes of hourly electricity data
        - Integrating weather data with electricity demand data
        - Designing visualizations that clearly communicate grid stress
        """
    )

    st.header("6. Proposed Dashboard")
    st.write(
        """
        The final Streamlit dashboard contains:

        1. **Grid Overview**
           - KPI cards
           - Recent demand and forecast trend

        2. **Real-time Grid Monitor**
           - Forecast vs actual demand
           - Forecast error over time
           - Rolling error trend

        3. **Weather Impact Analysis**
           - Temperature vs demand
           - Temperature vs forecast error
           - Daily MAPE trend

        4. **Map Dashboard**
           - Regional prediction error
           - Regional demand stress
        """
    )
